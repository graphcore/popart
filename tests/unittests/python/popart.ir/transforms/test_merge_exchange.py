# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Type

import popart._internal.ir as _ir
import popart.ir as pir
import popart.ir.ops as ops


def test_merge_exchange():
    ir = pir.Ir()
    main = ir.main_graph()
    with main:
        with pir.transforms.merge_exchange():
            x = ops.host_load(pir.h2d_stream((), pir.float32), "x")
            ops.host_store(pir.d2h_stream((), pir.float32), pir.constant(1.0))

    mg_ops = main._pb_graph.getOps()
    # Init and MultiExchange
    assert len(mg_ops) == 2
    assert mg_ops[0].opType() == "Init"
    assert mg_ops[1].opType() == "MultiExchange"


def test_merge_exchange_targeted():
    ir = pir.Ir()
    main = ir.main_graph()
    with main:
        ops.host_store(pir.d2h_stream((), pir.float32), pir.constant(1.0))

        with pir.transforms.merge_exchange():
            x = ops.host_load(pir.h2d_stream((), pir.float32), "x")
            ops.host_store(pir.d2h_stream((), pir.float32), pir.constant(1.0))

    mg_ops = main._pb_graph.getOps()
    # Init and MultiExchange
    assert len(mg_ops) == 3
    assert mg_ops[0].opType() == "HostStore"
    assert mg_ops[1].opType() == "Init"
    assert mg_ops[2].opType() == "MultiExchange"
