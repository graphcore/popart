# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import pytest

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


def test_merge_exchange_remote():
    ir = pir.Ir()
    main = ir.main_graph()
    with main:
        with pir.transforms.merge_exchange():
            buffer = pir.remote_buffer((), pir.float32, 1)
            x = ops.remote_load(buffer, 0, "x")
            ops.remote_store(buffer, 0, pir.constant(1.0))

    mg_ops = main._pb_graph.getOps()
    # Init and MultiExchange
    assert len(mg_ops) == 2
    assert mg_ops[0].opType() == "Init"
    assert mg_ops[1].opType() == "MultiExchange"


def test_io_tile_exchange():
    ir = pir.Ir()
    main = ir.main_graph()
    with main, pir.in_sequence(True):
        with pir.transforms.io_tile_exchange():
            buffer = pir.remote_buffer((), pir.float32, 1)
            x = ops.remote_load(buffer, 0, "x")
            ops.remote_store(buffer, 0, pir.constant(1.0))

    mg_ops = main._pb_graph.getOps()
    # Init and MultiExchange
    assert len(mg_ops) == 2
    assert mg_ops[0].opType() == "Init"
    assert mg_ops[1].opType() == "MultiExchange"


def test_io_tile_exchange_failed_verify():
    ir = pir.Ir()
    main = ir.main_graph()
    with main, pir.in_sequence(True):
        with pytest.raises(RuntimeError):
            with pir.transforms.io_tile_exchange():
                buffer = pir.remote_buffer((), pir.float32, 1)

                # These two exchange ops have a data dependency so cannot be merged
                x = ops.remote_load(buffer, 0, "x")
                ops.remote_store(buffer, 0, x)
