# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from typing import Optional
import popxl
import popart._internal.ir as _ir
from popxl.context import get_current_context, op_debug_context
from popxl.tensor import Tensor
from .utils import check_in_graph, check_tensor_ipu_and_tile_set


@op_debug_context
def cumsum(t: Tensor, dim: Optional[int] = 0) -> Tensor:
    """
    Performs the cumulative sum of the input elements along the given dimension `dim`.

    See also `Pytorch Tensor.cumsum <https://pytorch.org/docs/stable/generated/torch.Tensor.cumsum.html>`__, `Numpy cumsum <https://numpy.org/doc/stable/reference/generated/numpy.cumsum.html>`__.

    Args:
        t (Tensor):
            The input tensor.
        dim (int):
            The dimension to perform the operation over.
    Returns:
        Tensor:
            The result contains the cumulative sum of the elements of the input tensor along the dimension `dim`.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t=t)
    check_tensor_ipu_and_tile_set(t=t)

    settings = ctx._get_op_settings('cumsum')
    opid = _ir.OperatorIdentifier("ai.onnx", "CumSum", 11, _ir.NumInputs(2, 2),
                                  1)
    dim_t = popxl.constant(dim, dtype=popxl.int64, name="dim")
    op = pb_g.createConnectedOp_CumSumOp({
        0: t.id,
        1: dim_t.id
    }, {
        0: g._create_tensor_id("cumsum_out"),
    }, opid, False, False, settings)

    return Tensor._from_pb_tensor(op.outTensor(0))
