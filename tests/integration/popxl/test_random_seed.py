# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popxl
import popxl.ops as ops
from popxl import dtypes


def test_random_seed_setup():
    ir = popxl.Ir()
    main = ir.main_graph
    with main:
        seed_h2d = popxl.h2d_stream(shape=(2,), dtype=dtypes.uint32, name="seed_stream")
        seed = ops.host_load(seed_h2d, "seed")

        x = popxl.variable(0.0)
        x = ops.dropout(x, seed + 1, p=0.1)
        y = ops.dropout(x, seed + 2, p=0.7)

        y_d2h = popxl.d2h_stream(y.shape, y.dtype, name="y_stream")
        ops.host_store(y_d2h, y)

    replicas = 4
    ir.replication_factor = replicas
    parent_seed = 1984
    seed_tensors = popxl.create_seeds(parent_seed, replicas=replicas)

    session = popxl.Session(ir, "ipu_model")

    with session:
        _ = session.run({seed_h2d: seed_tensors})
