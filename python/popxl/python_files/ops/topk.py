# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
from popxl.context import get_current_context, op_debug_context
from popxl.tensor import Tensor
from .utils import check_in_graph, convert_optional_float
from typing import Tuple, Optional


@op_debug_context
def topk(
    t: Tensor,
    k: int,
    axis: int,
    largest: bool,
    sorted: bool,
    available_memory_proportion: Optional[float] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Retrieve the top-K largest or smallest elements along a specified axis.

    See also `PyTorch torch.topk <https://pytorch.org/docs/stable/generated/torch.topk.html?highlight=topk#torch.topk>`__, `ONNX TopK <https://github.com/onnx/onnx/blob/main/docs/Operators.md#TopK>`__.

    Args:
        t:
            Input tensor.
        k:
            The number of top elements to retrieve
        axis:
            Dimension on which to do the sort.
        largest:
            Whether to return the top-K largest or smallest elements.
        sorted:
            Whether to return the elements in sorted order.
        available_memory_proportion: Optional[float]
            The maximum proportion of available memory on each tile that this
            layer should consume temporarily during the course of the operation.
            This value is used by the grad operator only. In other words value
            of that parameter is irrelevant to the inference use case.
            Defaults to 1.0 if not set globally.
    Returns:
        Tuple[Tensor, Tensor]:
            A tuple of output values and indices.
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    check_in_graph(g, t=t)

    available_memory_proportion = convert_optional_float(available_memory_proportion)

    settings = ctx._get_op_settings("topk")
    opid = _ir.OperatorIdentifier("ai.onnx", "TopK", 1, _ir.NumInputs(1, 1), 2)
    op = pb_g.createConnectedOp_TopKOp(
        {0: t.id},
        {
            0: g._create_tensor_id("topk_out_value"),
            1: g._create_tensor_id("topk_out_indices"),
        },
        opid=opid,
        k=k,
        axis=axis,
        largest=largest,
        sorted=sorted,
        settings=settings,
        available_memory_proportion=available_memory_proportion,
    )

    return Tensor._from_pb_tensor(op.outTensor(0)), Tensor._from_pb_tensor(
        op.outTensor(1)
    )
