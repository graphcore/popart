# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Optional
from typing_extensions import Literal
import popart._internal.ir as _ir
from popxl import ReplicaGrouping
from popxl.context import get_current_context
from popxl.tensor import Tensor
from popxl.ops.utils import check_in_graph
from .collectives import _rearrange_input, _rearrange_output


def replicated_all_gather(
    t: Tensor,
    axis: int = 0,
    group: Optional[ReplicaGrouping] = None,
    output_shape: Literal["new_axis", "concat", "meta_shape", "auto"] = "auto",
) -> Tensor:
    """Gather a tensor across replicas such that the output tensor contains the values of the tensor from each replica.

    The shape of the output tensor is determined by the value of `output_shape`:

        - `new_axis`: the output shape is `(group.size, *t.shape)`
        - `concat`: the output shape has the same behavior as concat on `axis`
        - `meta_shape`: the output shape is `t.meta_shape`
        - `auto`: if the input has a meta-shape `meta_shape` is chosen, otherwise `concat`

    This op is auto-differentiable and it's corresponding grad op is an replicated_slice
    (except when `output_shape==meta_shape`).

    Args:
        t (Tensor): Tensor to be gathered.
        axis (int): Axis to gather and concatenate values when using 'concat' mode
        group (Optional[ReplicaGrouping]): Replicas to gather from. Defaults to All replicas.
        output_shape (str): see above for details. Choose 'new_axis', 'concat', 'meta_shape' or 'auto'.
    Returns:
        Tensor: Gathered tensor.
    Raises:
        ValueError: if `output_shape` is not one of 'new_axis', 'concat', 'meta_shape' or 'auto'.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t=t)
    group = g.ir.replica_grouping() if group is None else group

    if not isinstance(axis, int):
        raise ValueError(f"Axis should be an int. Type: {type(axis)}. Value: {axis}")

    if output_shape == "auto":
        output_shape = "meta_shape" if t.meta_shape else "concat"

    if output_shape != "meta_shape":
        t_input, concat_shape = _rearrange_input(t, axis)
        concat_shape[0] *= group.group_size
    else:
        t_input = t

    if output_shape == "new_axis":
        new_shape = (group.group_size, *t.shape)
    elif output_shape == "concat":
        new_shape = concat_shape
    elif output_shape == "meta_shape":
        new_shape = t.meta_shape
    else:
        raise ValueError(
            "output_shape must equal one of 'new_axis', 'concat', 'meta_shape', 'auto'. "
            f"Value: {output_shape}"
        )

    settings = ctx._get_op_settings("replicated_all_gathered")
    opid = _ir.OperatorIdentifier(
        "ai.graphcore", "ReplicatedAllGather", 1, _ir.NumInputs(1, 2), 1
    )
    out_info = _ir.TensorInfo(t.dtype._pb_dtype, new_shape)

    _op = pb_g.createConnectedOp_ReplicatedAllGatherOp(
        {0: t_input.id},
        {0: g._create_tensor_id(t_input.name + "_all_gathered")},
        opid,
        group._pb_replica_grouping,
        settings,
        out_info,
    )

    out = Tensor._from_pb_tensor(_op.outTensor(0))

    if output_shape != "meta_shape":
        # Reshape here as well so auto-diff works
        out = _rearrange_output(out, new_shape, axis)

    return out
