# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
import popart.ir as pir
import popart.ir.ops as ops
from popart.ir import dtypes
from utils import contains_op_of_type
import numpy as np


class TestDropout:
    def test_fn(self):
        ir = pir.Ir()
        g = ir.main_graph()

        with g:
            x = pir.variable(0)
            seed = pir.variable(np.array([32, 32]), dtype=dtypes.uint32)
            c = ops.dropout(x, seed, 0.3)
        assert len(g.get_tensors()) == 3
        assert len(g.get_variables()) == 2
        assert contains_op_of_type("Dropout", _ir.op.DropoutOp, g)
