# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from functools import partial
from typing import Any, List, Optional

import numpy as np
import torch
import torch_scatter

import popart
import popart_core

reductions = ["sum", "max", "min", "mul"]
dtypes = [torch.float32, torch.float16, torch.int]

reduction_map = {
    "sum": popart_core.ScatterReduction.Sum,
    "max": popart_core.ScatterReduction.Max,
    "min": popart_core.ScatterReduction.Min,
    "mul": popart_core.ScatterReduction.Mul,
    "none": popart_core.ScatterReduction.NoReduction,
}


def pytorch_scatter_reduce_supported():
    version_to_int = lambda version: int("".join(version.split(".")[:2]))
    current_version = version_to_int(torch.__version__)
    return current_version >= version_to_int("1.13.0")


def create_test_id(val):
    if isinstance(val, torch.dtype):
        return f"{val}".split(".")[-1]

    return val


def torch_scatter_reduce_to_torch(reduce: str) -> str:
    if reduce == "mul":
        return "prod"
    elif reduce == "max":
        return "amax"
    elif reduce == "min":
        return "amin"

    return reduce


def scatter_reduce_reference(
    src_group: List[torch.Tensor],
    index_group: List[torch.Tensor],
    dim: int = -1,
    out_group: Optional[List[torch.Tensor]] = None,
    dim_size: Optional[int] = None,
    reduce: str = "sum",
    group_size: int = 1,
    use_torch_scatter_as_reference: bool = True,
) -> List[torch.Tensor]:

    if reduce == "none":
        use_torch_scatter_as_reference = False
    elif not pytorch_scatter_reduce_supported() and not use_torch_scatter_as_reference:
        use_torch_scatter_as_reference = True

    assert len(src_group) == group_size
    assert len(index_group) == group_size

    results = []
    include_self = out_group is not None

    def prepare_input():
        if include_self:
            input = out_group[group_id]
        else:
            input_size = list(index.shape)
            input_size[dim] = dim_size
            if reduce == "mul":
                input = torch.ones(*input_size).float()
            else:
                input = torch.zeros(*input_size).float()

        return input

    for group_id in range(group_size):
        src = src_group[group_id]
        index = index_group[group_id]
        out = None
        if out_group is not None:
            out = out_group[group_id]

        if reduce == "none":
            results.append(
                torch.scatter(prepare_input(), dim=dim, index=index, src=src)
            )
        elif use_torch_scatter_as_reference:
            results.append(
                torch_scatter.scatter(
                    src, index, dim, out, dim_size=dim_size, reduce=reduce
                )
            )
        else:
            results.append(
                torch.scatter_reduce(
                    input=prepare_input(),
                    dim=dim,
                    index=index,
                    src=src,
                    reduce=torch_scatter_reduce_to_torch(reduce),
                    include_self=include_self,
                )
            )

    if group_size == 1:
        return results

    return [torch.stack(results)]


def scatter_reduce_reference_backward(
    d__o: torch.Tensor,
    src_group: List[torch.Tensor],
    index_group: List[torch.Tensor],
    dim: int = -1,
    out_group: Optional[List[torch.Tensor]] = None,
    dim_size: Optional[int] = None,
    reduce: str = "sum",
    group_size: int = 1,
    use_torch_scatter_as_reference: bool = True,
) -> List[torch.Tensor]:

    if reduce == "none":
        use_torch_scatter_as_reference = False
    elif not pytorch_scatter_reduce_supported() and not use_torch_scatter_as_reference:
        use_torch_scatter_as_reference = True

    for src_tensor in src_group:
        src_tensor.requires_grad_()

    with_intial_values = out_group is not None

    if with_intial_values and not use_torch_scatter_as_reference:
        for out_tensor in out_group:
            out_tensor.requires_grad_()

    ref_out = scatter_reduce_reference(
        src_group,
        index_group,
        dim=dim,
        out_group=out_group,
        dim_size=dim_size,
        reduce=reduce,
        group_size=group_size,
        use_torch_scatter_as_reference=use_torch_scatter_as_reference,
    )[0]

    ref_out.backward(d__o)

    def get_gradients(group):
        if group is None:
            return None
        gradients = [tensor.grad for tensor in group]
        if None in gradients:
            return None
        return gradients[0] if group_size == 1 else torch.stack(gradients)

    if with_intial_values:
        return [ref_out, get_gradients(src_group), get_gradients(out_group), d__o]

    return [ref_out, get_gradients(src_group), d__o]


def scatter_reduce_builder(
    builder: Any,
    src_group: List[torch.Tensor],
    index_group: List[torch.Tensor],
    dim: int = -1,
    out_group: Optional[List[torch.Tensor]] = None,
    dim_size: Optional[int] = None,
    reduce: str = "sum",
    group_size: int = 1,
    enable_index_broadcast: bool = True,
    backward: bool = False,
):

    with_intial_values = out_group is not None

    if group_size > 1:
        scatter_func = partial(
            builder.aiGraphcore.groupedscatterreduce, group_size=group_size
        )
        if dim >= 0:
            dim += 1
        src = torch.stack(src_group)
        index = torch.stack(index_group)
        if with_intial_values:
            initial = torch.stack(out_group)
    else:
        scatter_func = builder.aiGraphcore.scatterreduce
        src = src_group[0]
        index = index_group[0]
        if with_intial_values:
            initial = out_group[0]

    D = builder.addInputTensor(src.numpy())
    I = builder.addInputTensor(index.numpy().astype(np.uint32))
    input_tensors = [D, I]

    if with_intial_values:
        V = builder.addInputTensor(initial.numpy())
        input_tensors.append(V)

    out = scatter_func(
        input_tensors,
        axis=dim,
        axis_size=dim_size,
        reduction=reduction_map[reduce],
        enable_index_broadcast=int(enable_index_broadcast),
    )

    builder.addOutputTensor(out)

    out_tensors = [out]
    if backward:
        out_tensors.append(popart.reservedGradientPrefix() + D)
        if with_intial_values:
            out_tensors.append(popart.reservedGradientPrefix() + V)
        out_tensors.append(popart.reservedGradientPrefix() + out)

    return out_tensors
