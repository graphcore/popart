# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from typing import Iterable
import popart._internal.ir as _ir
import popxl
from popxl.context import get_current_context
from popxl.ops.utils import check_in_graph, check_tensor_ipu_and_tile_set


def sum_ts(ts: Iterable[popxl.Tensor]) -> popxl.Tensor:
    """
    Sum Tensors

    Args:
        ts (Iterable[Tensor]):
            Tensors to be summed.
    Returns:
        Tensor
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    ts = list(ts)

    check_in_graph(g, **{f"ts_{i}": t for i, t in enumerate(ts)})
    check_tensor_ipu_and_tile_set(**{f"ts_{i}": t for i, t in enumerate(ts)})

    settings = ctx._get_op_settings("sum")
    opid = _ir.OperatorIdentifier("ai.onnx", "Sum", 6, _ir.NumInputs(2, -1), 1)
    op = pb_g.createConnectedOp_SumOp(
        {i: t.id for i, t in enumerate(ts)},
        {
            0: g._create_tensor_id("sum_out"),
        },
        opid,
        settings,
    )

    return popxl.Tensor._from_pb_tensor(op.outTensor(0))


def test_decompose_sum():
    ir = popxl.Ir()
    with ir.main_graph:
        xs = [popxl.constant(1) for _ in range(4)]
        ys = [popxl.constant(1) for _ in range(4)]

        sum_ts((sum_ts(xs), *ys))

    assert len(ir.main_graph._pb_graph.getOps()) == 2

    popxl.transforms.decompose_sum(ir.main_graph)

    # A sum with N inputs decomposes to N Adds + 1 Init
    # First sum as 4 inputs, second sum as 5 inputs
    assert len(ir.main_graph._pb_graph.getOps()) == 5 + 6
