# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import popxl
from popxl import ops
import numpy as np


class TestAutoDiffCast:
    # This test was failing due to bug in popxl cast.
    # It was creating output tensor by using tensor id
    # instead of name. Resulting in autodiff grow grads
    # error having wrong tensor names. See T62127.
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
