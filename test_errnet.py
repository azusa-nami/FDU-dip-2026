import argparse
import sys
from os.path import join

import torch.backends.cudnn as cudnn
from torch.utils.data import Subset

import data.reflect_dataset as datasets
from engine import Engine
from options.errnet.train_options import TrainOptions
from util.distributed import cleanup_distributed, is_main_process


EVAL_DATASETS = {
    "ceilnet_table2": {
        "dataset_name": "testdata_table2",
        "path": "test/ceilnet_table2",
        "save_subdir": "CEILNet_table2",
    },
    "real20": {
        "dataset_name": "testdata_real",
        "path": "test/real20",
        "save_subdir": "real20",
        "max_long_edge": 512,
    },
    "postcard": {
        "dataset_name": "testdata_postcard",
        "path": "test/postcard",
        "save_subdir": "postcard",
    },
    "objects": {
        "dataset_name": "testdata_objects",
        "path": "test/objects",
        "save_subdir": "objects",
    },
    "wild": {
        "dataset_name": "testdata_wild",
        "path": "test/wild",
        "save_subdir": "wild",
    },
    "sir2_withgt": {
        "dataset_name": "testdata_sir2",
        "path": "test/sir2_withgt",
        "save_subdir": "sir2_withgt",
    },
}

TEST_DATASETS = {
    "internet": {
        "path": None,
        "save_subdir": "internet",
    },
}


def parse_test_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(list(EVAL_DATASETS.keys()) + list(TEST_DATASETS.keys()) + ["all", "custom"]),
        help="dataset to run; use 'all' to evaluate every dataset with ground truth",
    )
    parser.add_argument(
        "--data_root",
        default="./datasets/data",
        help="root directory for processed datasets",
    )
    parser.add_argument(
        "--input_dir",
        default="./datasets/raw_data/CEILNet/testdata_reflection_real",
        help="input directory for test-only datasets such as internet/custom images",
    )
    parser.add_argument(
        "--result_dir",
        default="./results",
        help="directory for saved outputs",
    )
    parser.add_argument(
        "--save_subdir",
        default=None,
        help="override output subdirectory name under result_dir",
    )
    parser.add_argument(
        "--max_long_edge",
        type=int,
        default=None,
        help="resize test images so the longest edge does not exceed this value",
    )
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return args


def shard_dataset(dataset, opt):
    if not getattr(opt, "distributed", False):
        return dataset
    indices = list(range(opt.rank, len(dataset), opt.world_size))
    return Subset(dataset, indices)


def build_eval_dataloader(opt, data_root, dataset_key, max_long_edge=None):
    spec = EVAL_DATASETS[dataset_key]
    dataset = datasets.CEILTestDataset(
        join(data_root, spec["path"]),
        max_long_edge=max_long_edge if max_long_edge is not None else spec.get("max_long_edge"),
    )
    dataset = shard_dataset(dataset, opt)
    dataloader = datasets.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=opt.nThreads,
        pin_memory=len(opt.gpu_ids) > 0,
    )
    return spec, dataloader


def build_test_dataloader(opt, dataset_key, input_dir, max_long_edge=None):
    if dataset_key == "custom":
        dataset = datasets.RealDataset(input_dir, max_long_edge=max_long_edge)
        save_subdir = "custom"
    else:
        spec = TEST_DATASETS[dataset_key]
        dataset = datasets.RealDataset(
            input_dir if spec["path"] is None else join(input_dir, spec["path"]),
            max_long_edge=max_long_edge,
        )
        save_subdir = spec["save_subdir"]

    dataset = shard_dataset(dataset, opt)
    dataloader = datasets.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=opt.nThreads,
        pin_memory=len(opt.gpu_ids) > 0,
    )
    return save_subdir, dataloader


def run_eval_dataset(engine, opt, cli_args, dataset_key, savedir):
    spec, dataloader = build_eval_dataloader(
        opt,
        cli_args.data_root,
        dataset_key,
        max_long_edge=cli_args.max_long_edge,
    )
    if is_main_process(opt):
        print("[i] evaluating {}".format(dataset_key))
    res = engine.eval(
        dataloader,
        dataset_name=spec["dataset_name"],
        savedir=savedir,
        sync_distributed=getattr(opt, "distributed", False),
    )
    if is_main_process(opt):
        print("{}: {}".format(dataset_key, res))
    return res


def main():
    cli_args = parse_test_args()
    try:
        option_parser = TrainOptions()
        option_parser.isTrain = False
        opt = option_parser.parse()
        opt.isTrain = False
        if opt.icnn_path is not None and not opt.resume:
            opt.resume = True
            if is_main_process(opt):
                print("[i] --icnn_path was provided; enabling checkpoint loading")
        if not opt.resume:
            raise ValueError("test_errnet.py requires a checkpoint. Pass -r with --icnn_path, or provide --resume.")

        cudnn.benchmark = len(opt.gpu_ids) > 0
        opt.no_log = True
        opt.display_id = 0
        opt.verbose = False

        engine = Engine(opt)

        if cli_args.dataset == "all":
            results = {}
            for dataset_key, spec in EVAL_DATASETS.items():
                if cli_args.save_subdir is None:
                    savedir = join(cli_args.result_dir, spec["save_subdir"])
                else:
                    savedir = join(cli_args.result_dir, cli_args.save_subdir, spec["save_subdir"])
                results[dataset_key] = run_eval_dataset(engine, opt, cli_args, dataset_key, savedir)
            if is_main_process(opt):
                print("all results: {}".format(results))
        elif cli_args.dataset in EVAL_DATASETS:
            spec = EVAL_DATASETS[cli_args.dataset]
            save_subdir = cli_args.save_subdir or spec["save_subdir"]
            run_eval_dataset(
                engine,
                opt,
                cli_args,
                cli_args.dataset,
                join(cli_args.result_dir, save_subdir),
            )
        else:
            default_save_subdir, dataloader = build_test_dataloader(
                opt,
                cli_args.dataset,
                cli_args.input_dir,
                max_long_edge=cli_args.max_long_edge,
            )
            save_subdir = cli_args.save_subdir or default_save_subdir
            engine.test(
                dataloader,
                savedir=join(cli_args.result_dir, save_subdir),
            )
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()

"""
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run torchrun --master_port=29501 --nproc_per_node=4 test_errnet.py \
    --name errnet_harder_ddp \
    --hyper \
    --dataset all \
    -r \
    --icnn_path checkpoints/errnet_harder_ddp/errnet_latest.pt \
    --feature_model_path /oldhome/zengyuqi/model/dinov3
"""
