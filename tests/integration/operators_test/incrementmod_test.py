# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
import torch
import pytest
from op_tester import op_tester


@pytest.mark.parametrize("dtype",
                         [np.float16, np.float32, np.uint32, np.int32])
@pytest.mark.parametrize("inplace", [False, True])
def test_incrementmod(op_tester, dtype, inplace):
    increment = 3.0
    modulus = 4.0

    d1 = np.array([5.0, -5.0, 10.0, 1.0, 2.0], dtype=dtype)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiGraphcore.incrementmod([i1], increment, modulus)
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        out = np.fmod((d1 + np.asarray(increment, dtype)),
                      np.asarray(modulus, dtype))
        return [out]

    if inplace:
        op_tester.setPatterns(['InPlace'], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'infer')
