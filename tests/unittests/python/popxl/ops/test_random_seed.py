# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
import popxl
import popxl.ops as ops

from utils import contains_op_of_type


class TestSplitRandomSeed:
    """Ensure random seed APIs are available."""

    def test_fn(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            seed = popxl.variable((1, 2), popxl.uint32)
            _, _ = ops.split_random_seed(seed)

        assert len(g.tensors) == 5
        assert len(g.variables) == 1
        assert contains_op_of_type("ModifyRandomSeed",
                                   _ir.op.ModifyRandomSeedOp, g)

    def test_many(self):
        ir = popxl.Ir()
        g = ir.main_graph

        with g:
            seed = popxl.variable((1, 2), popxl.uint32)
            seeds = ops.split_random_seed(seed, 5)
            assert len(seeds) == 5
