# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from .add import add, add_
from .pool import average_pool, max_pool
from .call import call, call_with_info, CallSiteInfo
from .cast import cast
from .concat import concat, concat_
from .conv import conv, PadType
from .code_copy import remote_code_load
from .roialign import roi_align
from .argminmax import argmin, argmax
from .interpolate import interpolate
from .detach import detach, detach_
from .div import div
from .dropout import dropout
from .dynamic_slice import dynamic_slice
from .dynamic_update import dynamic_update, dynamic_update_
from .equal import equal
from .exp import exp, exp_
from .log import log
from .onehot import onehot
from .topk import topk
from .fmod import fmod
from .gather import gather, tied_gather
from .gelu import gelu, gelu_
from .group_norm import group_norm, layer_norm
from .host_load import host_load
from .host_store import host_store
from .increment_mod import increment_mod, increment_mod_
from .init import init
from .io_tile_copy import io_tile_copy
from .ipu_copy import ipu_copy
from .l1 import l1
from .l2 import l2
from .lamb import lamb_square
from .logical_and import logical_and
from .logical_not import logical_not
from .logical_or import logical_or
from .logsum import logsum
from .logsumexp import logsumexp
from .matmul import matmul
from .max import max, maximum
from .mean import mean
from .median import median
from .min import min
from .mul import mul
from .negate import negate
from .negative_log_likelihood import nll_loss, nll_loss_with_softmax_grad
from .print_tensor import print_tensor
from .prod import prod
from .random_seed import split_random_seed
from .random import random_uniform, random_normal
from .relu import relu, relu_
from .repeat import repeat, repeat_with_info
from .reshape import reshape, reshape_, flatten, flatten_
from .remote_load import remote_load, remote_load_
from .remote_store import remote_store
from .scaled_add import scaled_add, scaled_add_
from .scatter import scatter
from .slice import slice, slice_
from .softmax import softmax
from .split import split
from .sqrt import sqrt
from .squeeze import squeeze
from .sub import sub
from .sum import sum
from .sumsquare import sumsquare
from .tanh import tanh
from .transpose import transpose, transpose_
from .where import where

from . import collectives
from . import var_updates

__all__ = [
    # add.py
    "add",
    "add_",
    # argminmax.py
    "argmax",
    "argmin",
    # pool.py
    "average_pool",
    "max_pool",
    # call.py
    "call",
    "call_with_info",
    "CallSiteInfo",
    # cast.py
    "cast",
    # concat.py
    "concat",
    "concat_",
    # conv.py
    "conv",
    "PadType",
    # roialign.py
    "roi_align",
    # interpolate.py,
    "interpolate",
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
    # equal.py
    "equal",
    # exp.py
    "exp",
    "exp_",
    # log.py
    "log",
    # onehot.py
    "onehot",
    # topk.py
    "topk",
    # fmod.py
    "fmod",
    # gather.py
    "gather",
    "tied_gather",
    # gelu.py
    "gelu",
    "gelu_",
    # group_norm.py
    "group_norm",
    "layer_norm",
    # host_load.py
    "host_load",
    # host_store.py
    "host_store",
    # increment_mod.py
    "increment_mod",
    "increment_mod_",
    # init.py
    "init",
    # io_tile_copy.py
    "io_tile_copy",
    # ipu_copy.py
    "ipu_copy",
    # l1.py
    "l1",
    # l2.py
    "l2",
    # lamb.py
    "lamb_square",
    # logical_and.py
    "logical_and",
    # logical_not.py
    "logical_not",
    # logical_or.py
    "logical_or",
    # logsum.py
    "logsum",
    # logsumexp.py
    "logsumexp",
    # matmul.py
    "matmul",
    # max.py
    "max",
    "maximum",
    # mean.py
    "mean",
    # median.py
    "median",
    # min.py
    "min",
    # mul.py
    "mul",
    # negative_log_likelihood.py
    "nll_loss",
    "nll_loss_with_softmax_grad",
    # negate.py
    "negate",
    # print_tensor.py
    "print_tensor",
    # prod.py
    "prod",
    # random_seed.py
    "split_random_seed",
    # random.py
    "random_uniform",
    "random_normal",
    # relu.py,
    "relu",
    "relu_",
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
    # repeat.py,
    "repeat",
    "repeat_with_info",
    # scaled_add.py
    "scaled_add",
    "scaled_add_",
    # scatter.py
    "scatter",
    # slice.py
    "slice",
    "slice_",
    # softmax.py
    "softmax",
    # split.py
    "split",
    # sqrt.py
    "sqrt",
    # squeeze.py
    "squeeze",
    # sub.py
    "sub",
    # sum.py
    "sum",
    # sumsquare.py
    "sumsquare",
    # tanh.py
    "tanh",
    # transpose.py
    "transpose",
    "transpose_",
    # where.py
    "where",
    # sub-package popxl.ops.collectives
    "collectives",
    # sub-package popxl.ops.var_updates
    "var_updates",
]
