# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import popart._internal.ir as _ir
import popart.ir as pir


def test_pipeline_stage_with():
    ir = pir.Ir()
    g = ir.main_graph

    with g:
        x = pir.variable(1)
        with pir.pipeline_stage(0):
            y = x + 1

    ops = g._pb_graph.getOpSchedule()
    assert len(ops) == 1
    op: _ir.Op = ops[0]
    assert isinstance(op, _ir.op.AddOp)
    assert op.hasPipelineStage()
    assert op.getPipelineStage() == 0
