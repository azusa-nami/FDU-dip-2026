from pathlib import Path
import time

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
    specs = {
        "real20": {"path": "real20", "size": 20, "max_long_edge": 512},
        "objects": {"path": "objects"},
        "postcard": {"path": "postcard"},
        "wild": {"path": "wild"},
    }

    loader_kwargs = {
        "batch_size": 1,
        "shuffle": False,
        "num_workers": opt.nThreads,
        "pin_memory": len(opt.gpu_ids) > 0,
    }
    eval_loaders = {}
    for name, spec in specs.items():
        dataset = datasets.CEILTestDataset(
            str(test_root / spec["path"]),
            size=spec.get("size"),
            max_long_edge=spec.get("max_long_edge"),
        )
        eval_loaders[name] = datasets.DataLoader(dataset, **loader_kwargs)
    return eval_loaders


def set_learning_rate(engine, lr):
    for optimizer in engine.model.optimizers:
        if is_main_process(engine.opt):
            print("[i] set learning rate to {}".format(lr))
        util.set_opt_param(optimizer, "lr", lr)


def build_schedulers(engine, opt):
    if opt.lr_policy == "manual":
        return []
    if opt.lr_policy == "cosine":
        return [
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, opt.nEpochs),
                eta_min=opt.min_lr,
            )
            for optimizer in engine.model.optimizers
        ]
    if opt.lr_policy == "step":
        return [
            torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=opt.lr_step_size,
                gamma=opt.lr_gamma,
            )
            for optimizer in engine.model.optimizers
        ]
    raise NotImplementedError("LR policy [{}] is not implemented".format(opt.lr_policy))


def step_schedulers(schedulers, opt):
    for scheduler in schedulers:
        if not getattr(scheduler.optimizer, "_opt_called", False):
            continue
        scheduler.step()
    if schedulers and is_main_process(opt):
        lr = schedulers[0].optimizer.param_groups[0]["lr"]
        print("[i] scheduler lr: {}".format(lr))


def set_synthesis_schedule(train_loader, opt, epoch):
    if opt.synthesis != "mixed":
        return

    ratio = (0.8, 0.2)
    synthetic_dataset = train_loader.dataset.datasets[0]
    synthetic_dataset.set_synthesis_mix(*ratio)
    if is_main_process(opt):
        print("[i] synthesis mix reflection2/advanced: {:.1f}/{:.1f}".format(*ratio))


def is_better_score(score, best_score, opt):
    if best_score is None:
        return True
    if opt.early_stop_metric == "LMSE":
        return score < best_score - opt.early_stop_min_delta
    return score > best_score + opt.early_stop_min_delta


def eval_validation_suite(engine, eval_loaders, opt):
    scores = {}
    metric = opt.early_stop_metric
    for dataset_name, loader in eval_loaders.items():
        meters = engine.eval(loader, dataset_name=dataset_name)
        if metric in meters.keys():
            scores[dataset_name] = meters[metric]

    if not scores:
        raise RuntimeError("Metric [{}] was not produced by validation datasets.".format(metric))

    score = sum(scores.values()) / len(scores)
    if is_main_process(opt):
        details = ", ".join("{}={:.4f}".format(name, value) for name, value in scores.items())
        print("[i] validation {} mean={:.4f} ({})".format(metric, score, details))
    return score, scores


def sync_stop_signal(engine, opt, epoch, should_stop):
    if not opt.distributed:
        return should_stop

    signal_path = Path(engine.basedir) / "early_stop_signal.txt"
    if is_main_process(opt):
        signal_path.write_text("{} {}\n".format(epoch, int(should_stop)))
        return should_stop

    while True:
        if signal_path.exists():
            parts = signal_path.read_text().strip().split()
            if len(parts) == 2 and int(parts[0]) == epoch:
                return bool(int(parts[1]))
        time.sleep(5)


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
        eval_loaders = build_eval_loaders(opt)
        engine = Engine(opt)
        schedulers = build_schedulers(engine, opt)
        best_val_score = None
        bad_eval_count = 0

        if opt.resume and is_main_process(opt):
            engine.eval(eval_loaders["real20"], dataset_name="real20")
        barrier(opt)

        engine.model.opt.lambda_gan = 0
        if opt.lr_policy == "manual":
            set_learning_rate(engine, 1e-4)

        while engine.epoch < opt.nEpochs:
            if engine.epoch == 20:
                engine.model.opt.lambda_gan = 0.01
            if opt.lr_policy == "manual" and engine.epoch == 30:
                set_learning_rate(engine, 5e-5)
            if opt.lr_policy == "manual" and engine.epoch == 40:
                set_learning_rate(engine, 1e-5)
            if engine.epoch == 45:
                if opt.lr_policy == "manual":
                    set_learning_rate(engine, 5e-5)
            if opt.lr_policy == "manual" and engine.epoch == 50:
                set_learning_rate(engine, 1e-5)

            set_synthesis_schedule(train_loader, opt, engine.epoch)
            engine.train(train_loader)
            step_schedulers(schedulers, opt)

            should_stop = False
            if engine.epoch % opt.eval_freq == 0:
                if is_main_process(opt):
                    val_score, _ = eval_validation_suite(engine, eval_loaders, opt)
                    if is_better_score(val_score, best_val_score, opt):
                        best_val_score = val_score
                        bad_eval_count = 0
                        print("[i] new best validation {}: {:.4f}".format(opt.early_stop_metric, val_score))
                        engine.model.save(label="best_{}_val".format(opt.early_stop_metric.lower()))
                    else:
                        bad_eval_count += 1
                        print("[i] validation did not improve: {}/{}".format(bad_eval_count, opt.early_stop_patience))
                    should_stop = opt.early_stop_patience > 0 and bad_eval_count >= opt.early_stop_patience

                should_stop = sync_stop_signal(engine, opt, engine.epoch, should_stop)
                if should_stop:
                    if is_main_process(opt):
                        print("[i] early stopping at epoch {}".format(engine.epoch))
                    break
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
