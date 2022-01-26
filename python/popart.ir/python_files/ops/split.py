# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import List, Union
import popart._internal.ir as _ir
from popart.ir.context import get_current_context, op_debug_context
from popart.ir.tensor import Tensor
from .utils import check_in_graph, handle_negative_axis


@op_debug_context
def split(t: Tensor, splits: Union[int, List[int]],
          axis: int = 0) -> List[Tensor]:
    """
    Splits a tensor on a given axis into a list of tensors.

    Args:
        t: Tensor
            Tensor to be split.
        splits: int or List[int]
            Either an int which specifies the number of splits or a list of ints specifing the length of each output tensor.
        axis: int (default 0)
            Which axis to split on
    Returns:
        out: List[Tensor]
            A list of tensors
    """

    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t=t)

    axis = handle_negative_axis(t, axis)

    if isinstance(splits, int):
        axis_len = t.shape[axis]
        if axis_len % splits != 0:
            raise ValueError(
                f"Split {splits} does not equally divide tensor axis {axis} of length {axis_len}."
            )
        splits = [axis_len // splits] * splits

    outputs_t = {
        i: g._create_tensor_id(f"{t.name}_split_{i}")
        for i in range(len(splits))
    }

    settings = ctx._get_op_settings('split')
    opid = _ir.OperatorIdentifier("ai.onnx", "Split", 2, _ir.NumInputs(1, 1),
                                  1)
    op = pb_g.createConnectedOp_SplitOp(
        {0: t.id},
        outputs_t,
        axis_=axis,
        split_=splits,
        opid=opid,
        settings=settings,
    )

    output = [
        Tensor._from_pb_tensor(op.outTensor(i)) for i in range(len(splits))
    ]

    return output
