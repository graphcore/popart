# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import numpy as np
import pytest
import popart
import torch

# pva is needed when calling session.getReport()
import pva  # pylint: disable=unused-import

# `import op_tester` requires adding to sys.path
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "operators_test"))

# pylint is disabled as op_tester is used as a fixture
from conftest import op_tester  # pylint: disable=unused-import


@pytest.mark.parametrize("expand_shape", ((2, 50, 3, 50), (2, 1, 3, 1)))
@pytest.mark.parametrize("inplace", (True, False))
def test_expandcast(op_tester, expand_shape, inplace):
    in_shape = (2, 1, 3, 1)

    x = np.random.randn(*in_shape).astype(np.float16)
    expand_shape_array = np.array(expand_shape, dtype=np.int32)

    def get_memory(include_pattern):
        def init_builder(builder):
            x_t = builder.addInputTensor(x)
            e_shape = builder.aiOnnx.constant(expand_shape_array)
            e_t = builder.aiOnnx.expand([x_t, e_shape])
            c_t = builder.aiOnnx.cast([e_t], "FLOAT")
            one_t = builder.aiOnnx.constant(np.array(1.0, dtype=np.float32))
            a_t = builder.aiOnnx.add([c_t, one_t])
            builder.addOutputTensor(a_t)
            return [
                a_t,
                popart.reservedGradientPrefix() + a_t,
                popart.reservedGradientPrefix() + x_t,
            ]

        def reference(ref_data):
            # Use float32 as earlier torch versions do not support sum_cpu for
            # float16.
            x_t = torch.tensor(x.astype(np.float32), requires_grad=True)
            c_t = x_t.expand(expand_shape)
            a_t = c_t + 1.0
            d__o = ref_data.getOutputTensorGrad(0)
            a_t.backward(torch.tensor(d__o))
            return [a_t, None, x_t.grad.to(torch.float16)]

        patterns = ["InPlace"] if inplace else []
        if include_pattern:
            patterns.append("ExpandCast")
        op_tester.setPatterns(patterns, enableRuntimeAsserts=False)

        session = op_tester.run(init_builder, reference, "train")

        report = session.getReport()

        total_mem = 0
        max_on_tile = 0
        for t in report.compilation.tiles:
            total_mem = total_mem + t.memory.total.includingGaps
            max_on_tile = max(max_on_tile, t.memory.total.includingGaps)
        print(f"total_mem: {total_mem}")
        print(f"max_on_tile: {max_on_tile}")

        return total_mem, max_on_tile

    tm0, max0 = get_memory(False)
    tm1, max1 = get_memory(True)

    if expand_shape == in_shape:
        assert tm0 == tm1
        assert max0 == max1
    else:
        assert tm0 > tm1
        assert max0 > max1
