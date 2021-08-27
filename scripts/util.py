# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import operator
from itertools import accumulate
from pathlib import Path


def get_project_source_dir():
    return Path(__file__).resolve().parents[1].resolve()


def bor(*args: int):
    """Bitwise or.

    Example:
        bor(0x01, 0x10) == 0x01 | 0x10

    Returns:
        int: Inputs.
    """
    return list(accumulate(args, operator.or_))[-1]
