# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
from popxl.context import get_current_context, op_debug_context
from popxl.tensor import Tensor
from .utils import check_in_graph, check_tensor_ipu_and_tile_set


@op_debug_context
def onehot(t: Tensor, num_classes: Tensor, values: Tensor,
           axis: int) -> Tensor:
    """
    Produces a one-hot tensor based on inputs.

    See also `ONNX OneHot <https://github.com/onnx/onnx/blob/main/docs/Operators.md#OneHot>`__.

    Args:
        t: Tensor
            Input tensor containing indices.
        num_classes: Tensor
            Scalar specifying the number of classes in one-hot tensor.
        values: Tensor
            The value used for filling locations specified in 't' input tensor
        axis: int
            Axis along which one-hot representation in added.
    Returns:
        out: Tensor
            Output tensor.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t=t, num_classes=num_classes, values=values)
    check_tensor_ipu_and_tile_set(t=t, num_classes=num_classes, values=values)

    settings = ctx._get_op_settings('onehot')
    opid = _ir.OperatorIdentifier("ai.onnx", "OneHot", 9, _ir.NumInputs(3, 3),
                                  1)
    op = pb_g.createConnectedOp_OnehotOp(
        {
            0: t.id,
            1: num_classes.id,
            2: values.id
        }, {0: g._create_tensor_id("onehot_out")},
        opid=opid,
        axis_=axis,
        settings=settings)

    return Tensor._from_pb_tensor(op.outTensor(0))
