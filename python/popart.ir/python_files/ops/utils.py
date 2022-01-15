# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Optional

import popart._internal.ir as _ir
from popart.ir.tensor import Tensor
from popart.ir.graph import Graph
from popart.ir.dtypes import dtype
from popart.ir.errors import UndefinedValue
from typing import List


def cast_if_needed(t: Tensor, data_type: dtype) -> Tensor:
    from popart.ir.ops.cast import cast
    if t.dtype != data_type:
        return cast(t, data_type)
    return t


def check_in_graph(graph: Graph, *args, **tensors: Tensor):
    """
    Checks if tensors are in graph and is a tensor. If not, raises a ValueError or TypeError.
    Specify all tensors using keyword augments to aid the error message.
    """
    #Python >=3.7 signature `check_in_graph(graph, *, **tensors: Tensor)`
    if len(args) > 0:
        raise Exception(
            "Internal error: please specify all tensors as keyword arguments.")

    for name, tensor in tensors.items():
        if not isinstance(tensor, Tensor):
            raise TypeError(
                f"The input `{name}` is not a Tensor. Type: {type(tensor)}. Value: {tensor}."
            )
        if tensor not in graph:
            raise ValueError(
                f"The input tensor `{name}` is not in the current Graph {graph.name}."
            )


def check_tensor_ipu_and_tile_set(*args, **tensors: Tensor):
    """
    Check that all tensors exist on the same IPUs and tile sets.
    If an tensor's IPU or tile set cannot be determined the check is skipped.
    Specify all tensors using keyword augments to aid the error message.
    """
    #Python >=3.7 signature `check_tensor_ipu_and_tile_set(*, **tensors: Tensor)`
    if len(args) > 0:
        raise Exception(
            "Internal error: please specify all tensors as keyword arguments.")

    def get_ipu_and_tile_set(t):
        try:
            return t._get_ipu_and_tile_set(raise_on_undefined_tile_set=False,
                                           raise_on_undefined_ipu=False)
        except UndefinedValue:
            return None, None

    ipus, tile_sets, names = zip(*[(*get_ipu_and_tile_set(tensor), name)
                                   for name, tensor in tensors.items()])

    ipus_and_names = zip(ipus, names)
    tile_sets_and_names = zip(tile_sets, names)

    ipus_and_names = [(ipu, name) for ipu, name in ipus_and_names
                      if ipu is not None and ipu != -1]
    tile_sets_and_names = [(tile_set, name)
                           for tile_set, name in tile_sets_and_names
                           if tile_set is not None and tile_set != 'undefined']

    if len(ipus_and_names) > 1:
        ipu_0, name_0 = ipus_and_names[0]
        for ipu, name in ipus_and_names[1:]:
            if ipu != ipu_0:
                raise ValueError(
                    f"The input tensors `{name_0}` and `{name}` are located on different IPUs: {ipu_0} != {ipu}."
                )

    if len(tile_sets_and_names) > 1:
        tile_set_0, name_0 = tile_sets_and_names[0]
        for tile_set, name in tile_sets_and_names[1:]:
            if tile_set != tile_set_0:
                raise ValueError(
                    f"The input tensors `{name_0}` and `{name}` are located on different tile sets: {tile_set_0} != {tile_set}."
                )


def handle_negative_axis(t: Tensor, axis: int) -> int:
    return len(t.shape) + axis if axis < 0 else axis


def convert_optional_float(v: Optional[float]):
    return _ir.OptionalFloat(v) if v is not None else _ir.OptionalFloat()


def convert_optional_int(v: Optional[int]):
    return _ir.OptionalInt(v) if v is not None else _ir.OptionalInt()


def convert_optional_dtype(dt: Optional[dtype]):
    return _ir.OptionalDataType(
        dt._pb_dtype) if dt is not None else _ir.OptionalDataType()
