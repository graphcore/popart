# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from .all_reduce import (all_reduce, all_reduce_identical_inputs,
                         all_reduce_identical_grad_inputs)
from .collectives import CommGroup, CommGroupType
from .replicated_all_gather import replicated_all_gather
from .replicated_all_reduce import (replicated_all_reduce,
                                    replicated_all_reduce_)
from .replicated_reduce_scatter import replicated_reduce_scatter, replica_sharded_slice

__all__ = [
    # all_reduce.py
    "all_reduce",
    "all_reduce_identical_inputs",
    "all_reduce_identical_grad_inputs",
    # collectives.py
    "CommGroup",
    "CommGroupType",
    # replicated_all_gather.py
    "replicated_all_gather",
    # replicated_all_reduce.py
    "replicated_all_reduce",
    "replicated_all_reduce_",
    # replicated_reduce_scatter.py
    "replicated_reduce_scatter",
    "replica_sharded_slice"
]
