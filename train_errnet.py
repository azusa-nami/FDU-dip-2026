from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data.distributed import DistributedSampler

import data.reflect_dataset as datasets
import util.util as util
from data.image_folder import read_fns
from engine import Engine
from options.errnet.train_options import TrainOptions
from util.distributed import barrier, cleanup_distributed, is_main_process


def build_train_loader(opt):
    train_root = Path(opt.data_root) / "train"
    synthetic_dir = train_root / "synthetic_voc"
    real_dir = train_root / "real89"

    synthetic = datasets.CEILDataset(
        str(synthetic_dir),
        read_fns("VOC2012_224_train_png.txt"),
        size=opt.max_dataset_size,
        enable_transforms=True,
        low_sigma=opt.low_sigma,
        high_sigma=opt.high_sigma,
        low_gamma=opt.low_gamma,
        high_gamma=opt.high_gamma,
        synthesis=opt.synthesis,
    )
    real = datasets.CEILTestDataset(str(real_dir), enable_transforms=True)
    fusion = datasets.FusionDataset([synthetic, real], [0.7, 0.3])

    sampler = None
    shuffle = not opt.serial_batches
    if opt.distributed:
        sampler = DistributedSampler(fusion, shuffle=shuffle, drop_last=False)
        shuffle = False

    return datasets.DataLoader(
        fusion,
        batch_size=opt.batchSize,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=opt.nThreads,
        pin_memory=len(opt.gpu_ids) > 0,
    )


def build_eval_loaders(opt):
    test_root = Path(opt.data_root) / "test"
    ceilnet = datasets.CEILTestDataset(str(test_root / "ceilnet_table2"))
    real20 = datasets.CEILTestDataset(str(test_root / "real20"), size=20, max_long_edge=512)

    loader_kwargs = {
        "batch_size": 1,
        "shuffle": False,
        "num_workers": opt.nThreads,
        "pin_memory": len(opt.gpu_ids) > 0,
    }
    return (
        datasets.DataLoader(ceilnet, **loader_kwargs),
        datasets.DataLoader(real20, **loader_kwargs),
    )


def set_learning_rate(engine, lr):
    for optimizer in engine.model.optimizers:
        if is_main_process(engine.opt):
            print("[i] set learning rate to {}".format(lr))
        util.set_opt_param(optimizer, "lr", lr)


def set_synthesis_schedule(train_loader, opt, epoch):
    if opt.synthesis != "mixed":
        return

    if epoch < 0.7 * opt.nEpochs:
        ratio = (0.8, 0.2)
    else:
        ratio = (0.6, 0.4)

    synthetic_dataset = train_loader.dataset.datasets[0]
    synthetic_dataset.set_synthesis_mix(*ratio)
    if is_main_process(opt):
        print("[i] synthesis mix reflection2/advanced: {:.1f}/{:.1f}".format(*ratio))


def main():
    opt = TrainOptions().parse()
    cudnn.benchmark = len(opt.gpu_ids) > 0

    if opt.debug:
        opt.display_id = 0
        opt.display_freq = 20
        opt.print_freq = 20
        opt.nEpochs = 2
        opt.max_dataset_size = 100
        opt.no_log = False
        opt.nThreads = 0
        opt.serial_batches = True
        opt.no_flip = True

    try:
        train_loader = build_train_loader(opt)
        eval_ceilnet_loader, eval_real20_loader = build_eval_loaders(opt)
        engine = Engine(opt)

        if opt.resume and is_main_process(opt):
            engine.eval(eval_ceilnet_loader, dataset_name="testdata_table2")
        barrier(opt)

        engine.model.opt.lambda_gan = 0
        set_learning_rate(engine, 1e-4)

        while engine.epoch < opt.nEpochs:
            if engine.epoch == 20:
                engine.model.opt.lambda_gan = 0.01
            if engine.epoch == 30:
                set_learning_rate(engine, 5e-5)
            if engine.epoch == 40:
                set_learning_rate(engine, 1e-5)
            if engine.epoch == 45:
                ratio = [0.5, 0.5]
                if is_main_process(opt):
                    print("[i] adjust fusion ratio to {}".format(ratio))
                train_loader.dataset.fusion_ratios = ratio
                set_learning_rate(engine, 5e-5)
            if engine.epoch == 50:
                set_learning_rate(engine, 1e-5)

            set_synthesis_schedule(train_loader, opt, engine.epoch)
            engine.train(train_loader)

            if is_main_process(opt) and engine.epoch % 5 == 0:
                engine.eval(eval_ceilnet_loader, dataset_name="testdata_table2")
                engine.eval(eval_real20_loader, dataset_name="testdata_real20")
            barrier(opt)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()

"""
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run torchrun --nproc_per_node=4 train_errnet.py \
    --name errnet_harder_ddp \
    --hyper
"""
