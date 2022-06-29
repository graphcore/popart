# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import pytest

import popxl
import popxl.ops as ops


def test_merge_exchange():
    ir = popxl.Ir()
    main = ir.main_graph
    with main:
        with popxl.transforms.merge_exchange():
            _ = ops.host_load(popxl.h2d_stream((), popxl.float32), "x")
            ops.host_store(popxl.d2h_stream((), popxl.float32),
                           popxl.constant(1.0))

    mg_ops = main._pb_graph.getOps()
    # Init and MultiExchange
    assert len(mg_ops) == 2
    assert mg_ops[0].opType() == "Init"
    assert mg_ops[1].opType() == "MultiExchange"


def test_merge_exchange_targeted():
    ir = popxl.Ir()
    main = ir.main_graph
    with main:
        ops.host_store(popxl.d2h_stream((), popxl.float32),
                       popxl.constant(1.0))

        with popxl.transforms.merge_exchange():
            _ = ops.host_load(popxl.h2d_stream((), popxl.float32), "x")
            ops.host_store(popxl.d2h_stream((), popxl.float32),
                           popxl.constant(1.0))

    mg_ops = main._pb_graph.getOps()
    # Init and MultiExchange
    assert len(mg_ops) == 3
    assert mg_ops[0].opType() == "HostStore"
    assert mg_ops[1].opType() == "Init"
    assert mg_ops[2].opType() == "MultiExchange"


def test_merge_exchange_remote():
    ir = popxl.Ir()
    main = ir.main_graph
    with main:
        with popxl.transforms.merge_exchange():
            buffer = popxl.remote_buffer((), popxl.float32, 1)
            _ = ops.remote_load(buffer, 0, "x")
            ops.remote_store(buffer, 0, popxl.constant(1.0))

    mg_ops = main._pb_graph.getOps()
    # Init and MultiExchange
    assert len(mg_ops) == 2
    assert mg_ops[0].opType() == "Init"
    assert mg_ops[1].opType() == "MultiExchange"


def test_io_tile_exchange():
    ir = popxl.Ir()
    main = ir.main_graph
    with main, popxl.in_sequence(True):
        with popxl.transforms.io_tile_exchange():
            buffer = popxl.remote_buffer((), popxl.float32, 1)
            _ = ops.remote_load(buffer, 0, "x")
            ops.remote_store(buffer, 0, popxl.constant(1.0))

    mg_ops = main._pb_graph.getOps()
    # Init and MultiExchange
    assert len(mg_ops) == 2
    assert mg_ops[0].opType() == "Init"
    assert mg_ops[1].opType() == "MultiExchange"


def test_io_tile_exchange_failed_verify():
    ir = popxl.Ir()
    main = ir.main_graph
    with main, popxl.in_sequence(True):
        with pytest.raises(RuntimeError):
            with popxl.transforms.io_tile_exchange():
                buffer = popxl.remote_buffer((), popxl.float32, 1)

                # These two exchange ops have a data dependency so cannot be merged
                x = ops.remote_load(buffer, 0, "x")
                ops.remote_store(buffer, 0, x)
