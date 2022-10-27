# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import pytest
import popxl
import popxl.ops as ops
from popxl.transforms import autodiff
import numpy as np


class TestReplicatedAllGather:
    def test_output_shape_new_axis(self):
        replication = 4

        x_np = np.random.rand(replication, 4, 3)

        ir = popxl.Ir(replication=replication)
        rg = ir.replica_grouping(group_size=replication)
        main = ir.main_graph

        with main:
            x = popxl.variable(
                x_np, popxl.float32, name="x", replica_grouping=rg.transpose()
            )

            def replicated_all_gather(x):
                return ops.collectives.replicated_all_gather(
                    x, axis=0, output_shape="new_axis", group=rg
                )

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

        y_np = outputs[y_d2h][0]
        dx_np = outputs[dx_d2h]

        np.testing.assert_almost_equal(y_np, x_np)
        np.testing.assert_almost_equal(dx_np, x_np)

    @pytest.mark.parametrize("output_shape", ["concat", "auto"])
    def test_output_shape_concat(self, output_shape):
        # Concat on axis == 1
        replication = 4

        x_np = np.random.rand(replication, 4, 3)

        ir = popxl.Ir(replication=replication)
        rg = ir.replica_grouping(group_size=replication)
        main = ir.main_graph

        with main:
            x = popxl.variable(
                x_np, popxl.float32, name="x", replica_grouping=rg.transpose()
            )

            def replicated_all_gather(x):
                return ops.collectives.replicated_all_gather(
                    x, axis=1, output_shape=output_shape, group=rg
                )

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

        y_np = outputs[y_d2h][0]
        dx_np = outputs[dx_d2h]

        y_true = np.concatenate(
            [a.squeeze() for a in np.split(x_np, replication, axis=0)], axis=1
        )

        np.testing.assert_almost_equal(y_np, y_true)
        np.testing.assert_almost_equal(dx_np, x_np)

    @pytest.mark.parametrize("output_shape", ["meta_shape", "auto"])
    def test_output_shape_meta_shape(self, output_shape):
        replication = 4

        x_np = np.random.rand(replication, 3, 2)

        ir = popxl.Ir(replication=replication)
        rg = ir.replica_grouping(group_size=replication)
        main = ir.main_graph

        with main, popxl.in_sequence(True):
            _, x = popxl.replica_sharded_variable(
                x_np, popxl.float32, name="x", replica_grouping=rg
            )

            def replicated_all_gather(x):
                return ops.collectives.replicated_all_gather(
                    x, axis=0, output_shape=output_shape, group=rg
                )

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

        y_np = outputs[y_d2h][0]
        dx_np = outputs[dx_d2h].reshape(x_np.shape)

        np.testing.assert_almost_equal(y_np, x_np)
        np.testing.assert_almost_equal(dx_np, x_np)
