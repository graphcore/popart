# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import popart._internal.ir as _ir
import popart.ir as pir
import popart.ir.ops as ops

from utils import contains_op_of_type


class TestSplitRandomSeed:
    """Ensure random seed APIs are available."""

    def test_fn(self):
        ir = pir.Ir()
        g = ir.main_graph()

        with g:
            seed = pir.variable((1, 2), pir.uint32)
            new_seed, other_seed = ops.split_random_seed(seed)

        assert len(g.get_tensors()) == 5
        assert len(g.get_variables()) == 1
        assert contains_op_of_type("ModifyRandomSeed",
                                   _ir.op.ModifyRandomSeedOp, g)

    def test_many(self):
        ir = pir.Ir()
        g = ir.main_graph()

        with g:
            seed = pir.variable((1, 2), pir.uint32)
            seeds = ops.split_random_seed(seed, 5)
            assert len(seeds) == 5
