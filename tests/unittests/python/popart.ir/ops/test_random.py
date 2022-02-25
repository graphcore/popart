# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
import popart.ir as pir
import popart.ir.ops as ops
from popart.ir import dtypes
from utils import contains_op_of_type
import numpy as np


class TestRandomUniform:
    def test_fn(self):
        ir = pir.Ir()
        g = ir.main_graph

        with g:
            seed = pir.variable(np.array([32, 32]), dtype=dtypes.uint32)
            x = ops.random_uniform(seed, (2, 2))
        assert len(g.tensors) == 2
        assert len(g.variables) == 1
        assert contains_op_of_type("RandomUniform", _ir.op.RandomUniformOp, g)


class TestRandomNormal:
    def test_fn(self):
        ir = pir.Ir()
        g = ir.main_graph

        with g:
            seed = pir.variable(np.array([32, 32]), dtype=dtypes.uint32)
            x = ops.random_normal(seed, (2, 2))
        assert len(g.tensors) == 2
        assert len(g.variables) == 1
        assert contains_op_of_type("RandomNormal", _ir.op.RandomNormalOp, g)
