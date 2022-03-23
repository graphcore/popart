# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from typing import Tuple
import numpy as np

MAX_UINT64 = 2**64 - 1


def create_seeds(seed: int,
                 offset: int = 0,
                 batches_per_step: int = 1,
                 gradient_accumulation_factor: int = 1,
                 replicas: int = 1) -> np.ndarray:
    """
    Create seed tensors from a parent seed. A seed tensor is a uint64 represented as two uint32s.
    If required, it creates multiple seed tensors for a the number of `batches_per_step`,
    `gradient_accumulation_factor` and `replicas` as needed.


    Args:
        seed (int):
            Initial seed
        offset (int):
            Offset `seed` by given amount
        batches_per_step (int, optional):
            Number of batches per step. Defaults to 1.
        replicas (int, optional):
            Number of model replications. Defaults to 1.
        gradient_accumulation_factor (int, optional):
            Gradient accumulation factor of model. Defaults to 1.

    Returns:
        np.ndarray: seed_tensors
    """
    seeds = np.random.default_rng(seed + offset).integers(
        0,
        MAX_UINT64,
        endpoint=True,
        dtype='uint64',
        size=batches_per_step * gradient_accumulation_factor * replicas)
    seed_tensors = np.stack([
        np.array(uint64_to_two_uint32(int(seed)), dtype='uint32')
        for seed in seeds
    ])
    seed_tensors = np.squeeze(
        seed_tensors.reshape(batches_per_step, gradient_accumulation_factor,
                             replicas, 2))
    return seed_tensors


def uint64_to_two_uint32(num: int) -> Tuple[int, int]:
    """Convert uint64 to two uint32s."""
    num_bin = f"{num:064b}"
    if num < 0 or len(num_bin) != 64:
        raise ValueError(
            f"Number is not positive or can be represented as a uint64: {num}")
    a = int(num_bin[:32], 2)
    b = int(num_bin[32:], 2)
    return a, b


def two_uint32_to_uint64(a: int, b: int) -> int:
    """Convert two uint32 to uint64."""
    a_bin = f"{a:032b}"
    b_bin = f"{b:032b}"
    if len(a_bin) != 32 or len(b_bin) != 32 or a < 0 or b < 0:
        raise ValueError(
            f"Either `a` or `b` are not positive or cannot be represented as a uint32: {a}, {b}"
        )
    num = int(a_bin + b_bin, 2)
    return num
