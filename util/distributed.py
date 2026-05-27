import os

import torch
import torch.distributed as dist


def init_distributed(opt):
    opt.rank = int(os.environ.get("RANK", "0"))
    opt.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    opt.world_size = int(os.environ.get("WORLD_SIZE", "1"))
    opt.distributed = opt.world_size > 1

    if opt.distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed training requires CUDA devices.")
        torch.cuda.set_device(opt.local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        opt.device = torch.device("cuda", opt.local_rank)
        opt.gpu_ids = [opt.local_rank]
    elif len(opt.gpu_ids) > 0:
        torch.cuda.set_device(opt.gpu_ids[0])
        opt.device = torch.device("cuda", opt.gpu_ids[0])
    else:
        opt.device = torch.device("cpu")

    return opt


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(opt=None):
    return opt is None or getattr(opt, "rank", 0) == 0


def barrier(opt=None):
    if opt is not None and getattr(opt, "distributed", False):
        dist.barrier()
