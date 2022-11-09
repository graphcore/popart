# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from typing import Type
import popxl
import popxl.ops as ops
import numpy as np
import popart._internal.ir as _ir


def _contains_op_of_type(
    opType: str, PbOpType: Type[_ir.Op], graph: popxl.Graph
) -> bool:
    pb_g = graph._pb_graph

    for pb_op in pb_g.getOps():
        if pb_op.opType() == opType and isinstance(pb_op, PbOpType):
            return True
    return False


def test_pattern():
    """Test that after a session is created with an IR that contains a ConvTranspose op,
    the ConvTransposePattern replaces it with a ConvFlipWeightsOp and a ConvOp"""
    batch_size = 1
    in_channel = 3
    out_channel = 3
    height = 50
    width = 50
    h_kernel = 3
    w_kernel = 3

    inputs = np.random.rand(batch_size, in_channel, height, width)
    weights = np.random.rand(out_channel, in_channel, h_kernel, w_kernel)
    ir = popxl.Ir()

    with ir.main_graph:
        t = popxl.variable(inputs, name="t")
        weights_t = popxl.variable(weights, name="weights")

        o = ops.conv_transpose(t, weights_t, groups=1)
        # print tensor to avoid alias zero copy disabling the op
        ops.print_tensor(o)

    # Creating the session will run the conv transpose pattern
    session = popxl.Session(ir)

    new_ir = session.ir_
    graph = new_ir.main_graph

    assert not _contains_op_of_type(
        "ConvTranspose", _ir.op.ConvTransposeOp, graph
    ), "New IR should not contain ConvTransposeOp"
    assert _contains_op_of_type(
        "Conv", _ir.op.ConvOp, graph
    ), "New IR should contain ConvOp"

    # ConvFlipWeightsOp is not exposed to check for its opType manually:
    op_types = {pb_op.opType() for pb_op in graph._pb_graph.getOps()}
    assert "ConvFlipWeights" in op_types
