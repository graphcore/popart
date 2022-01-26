# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import TYPE_CHECKING

from .add import add
from .call import call, call_with_info, SubgraphOpInfo
from .cast import cast
from .detach import detach, detach_
from .div import div
from .dropout import dropout
from .dynamic_slice import dynamic_slice
from .dynamic_update import dynamic_update, dynamic_update_
from .gather import gather, tied_gather
from .gelu import gelu
from .group_norm import group_norm, layer_norm
from .host_load import host_load
from .host_store import host_store
from .increment_mod import increment_mod, increment_mod_
from .init import init
from .io_tile_copy import io_tile_copy
from .ipu_copy import ipu_copy
from .lamb import lamb_square
from .logical_and import logical_and
from .logical_not import logical_not
from .logical_or import logical_or
from .matmul import matmul, SerialiseMode
from .mul import mul
from .negate import negate
from .negative_log_likelihood import nll_loss_with_softmax_grad
from .print_tensor import print_tensor
from .random import random_uniform, random_normal
from .repeat import repeat, repeat_with_info
from .reshape import reshape, reshape_, flatten, flatten_
from .remote_load import remote_load, remote_load_
from .remote_store import remote_store
from .scaled_add import scaled_add, scaled_add_
from .scatter import scatter
from .slice import slice, slice_
from .softmax import softmax
from .split import split
from .squeeze import squeeze
from .sub import sub
from .transpose import transpose, transpose_
from .where import where

from . import collectives
from . import var_updates

__all__ = [
    # add.py
    "add",
    # call.py
    "call",
    "call_with_info",
    "SubgraphOpInfo",
    # cast.py
    "cast",
    # detach.py
    "detach",
    "detach_",
    # div.py
    "div",
    # dropout.py
    "dropout",
    # dynamic_slice.py
    "dynamic_slice",
    # dynamic_update.py
    "dynamic_update",
    "dynamic_update_",
    # gather.py
    "gather",
    "tied_gather",
    # gelu.py
    "gelu",
    # group_norm.py
    "group_norm",
    "layer_norm",
    # host_load.py
    "host_load",
    # increment_mod.py
    "increment_mod",
    "increment_mod_",
    # init.py
    "init",
    # io_tile_copy.py
    "io_tile_copy",
    # ipu_copy.py
    "ipu_copy",
    # lamb.py
    "lamb_square",
    # logical_and.py
    "logical_and",
    # logical_not.py
    "logical_not",
    # logical_or.py
    "logical_or",
    # repeat.py,
    "repeat",
    "repeat_with_info",
    # matmul.py
    "matmul",
    "SerialiseMode",
    # mul.py
    "mul",
    # negative_log_likelihood.py
    "nll_loss_with_softmax_grad",
    # negate.py
    "negate",
    # print_tensor.py
    "print_tensor",
    # random.py
    "random_uniform",
    "random_normal",
    # reshape.py
    "reshape",
    "reshape_",
    "flatten",
    "flatten_",
    # remote_load.py
    "remote_load",
    "remote_load_",
    # remote_store.py
    "remote_store",
    # scaled_add.py
    "scaled_add",
    "scaled_add_",
    # slice.py
    "slice",
    "slice_",
    # softmax.py
    "softmax",
    # split.py
    "split",
    # squeeze.py
    "squeeze",
    # sub.py
    "sub",
    # transpose.py
    "transpose",
    "transpose_",
    # where.py
    "where",
    # sub-package popart.ir.ops.collectives
    "collectives",
    # sub-package popart.ir.ops.var_updates
    "var_updates",
]
