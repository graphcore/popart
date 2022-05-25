# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import popxl
from popxl import ops
import numpy as np


class TestAutoDiffCast:
    def test_auto_diff_cast(self):
        ir = popxl.Ir()
        main = ir.main_graph

        def layer(x: popxl.Tensor):
            a = ops.cast(x, x.dtype)
            x = x * a
            return x

        with main:
            x = popxl.variable(np.arange(10).astype('float32'), name='x')

            fwd_graph = ir.create_graph(layer, x.spec)
            grad_info = popxl.transforms.autodiff(fwd_graph)

            # Add following code so the graph is not optimised out.
            fwd_info = ops.call_with_info(fwd_graph, x)
            dx = ops.call(grad_info.graph,
                          popxl.constant(np.ones((10)), popxl.float32),
                          inputs_dict=grad_info.inputs_dict(fwd_info))

        with popxl.Session(ir, 'ipu_model') as session:
            output = session.run()
