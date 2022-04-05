# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import numpy as np
import popxl
import popxl.ops as ops
import popart._internal.ir as _ir
from utils import contains_op_of_type


def test_equal_op():
    ir = popxl.Ir()
    mg = ir.main_graph

    shape = (2, 4)

    with mg:
        a = popxl.variable(np.ones(shape))
        b = popxl.variable(np.ones(shape))
        c = ops.equal(a, b)

    assert c.shape == shape
    assert c.dtype == popxl.dtypes.bool
    assert len(mg.tensors) == 3
    assert contains_op_of_type("Equal", _ir.op.EqualOp, mg)
