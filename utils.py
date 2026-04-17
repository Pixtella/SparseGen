import torch
import random
import numpy as np
import torch.distributed as dist
import os
# ----------------------------------------------------------------------
#  DDP utils
# ----------------------------------------------------------------------

def setup_ddp():
    """Initialize torch.distributed using env vars set by torchrun."""
    if dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return device, local_rank

def is_main_process() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0

def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

def set_seed(seed = 0):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def num_to_groups(num: int, divisor: int) -> list:
    """
    Splits a number into groups of a given divisor.
    Args:
        num (int): The number to split.
        divisor (int): The size of each group.
    Returns:
        list: A list containing the sizes of the groups.
    """
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr
    
