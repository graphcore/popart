# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import popxl
import popxl.ops as ops
from popxl.transforms import autodiff
import numpy as np


class TestReplicatedSlice:
    def test_replicated_slice(self):
        replication = 4

        x_np = np.random.rand(replication, 4, 3)

        ir = popxl.Ir(replication=replication)
        rg = ir.replica_grouping(group_size=replication)
        main = ir.main_graph

        with main:
            x = popxl.variable(x_np, popxl.float32, name="x")

            def replicated_all_gather(x):
                return ops.collectives.replicated_slice(x, group=rg)

            graph = ir.create_graph(replicated_all_gather, x)
            dgraph = autodiff(graph)

            call_info = ops.call_with_info(graph, x)
            (y,) = call_info.outputs

            (dx,) = ops.call(dgraph.graph, y)

            y_d2h = popxl.d2h_stream(y.shape, y.dtype, name="y_stream")
            ops.host_store(y_d2h, y)

            dx_d2h = popxl.d2h_stream(dx.shape, dx.dtype, name="dx_stream")
            ops.host_store(dx_d2h, dx)

        with popxl.Session(ir, "ipu_model") as session:
            outputs = session.run()

        y_np = outputs[y_d2h]
        dx_np = outputs[dx_d2h][0]

        np.testing.assert_almost_equal(y_np, x_np)
        np.testing.assert_almost_equal(dx_np, x_np)
