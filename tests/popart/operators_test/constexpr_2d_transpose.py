import numpy as np
import pytest
import popart
import torch
from op_tester import op_tester


def test_constexpr_transpose_2D(op_tester):
    """
    In this test, 25 constexpr 2D transpose cases are run. 
    They are all run in the same compute graph, to avoid
    repeated graph generation overhead.
    """

    # testing all the edge cases when BS=16 (blocksize,
    # in blocking algorithm used to prevent cache misses)
    D0s = []
    D1s = []

    # we use fp16 for all odd test cases, fp16 for all even cases
    counter = 0

    # the two dimensions of the tensors are d0, d1,
    # we consider 25 combinations
    for d0 in [1, 15, 16, 17, 48]:
        for d1 in [1, 15, 16, 17, 32]:
            nptype = np.float16
            if counter % 2 == 0:
                nptype = np.float32
            D0s.append(np.random.rand(d0, d1).astype(nptype))
            D1s.append(np.random.rand(d1, d0).astype(nptype))
            counter += 1

    def init_builder(builder):
        outputs = []
        for i, D0 in enumerate(D0s):
            # i1 = builder.aiOnnx.constant(D0)
            i1 = builder.addInputTensor(D0)
            c = builder.aiOnnx.constant(D1s[i])
            ct = builder.aiOnnx.transpose([c], [1, 0])
            o = builder.aiOnnx.add([i1, ct])
            builder.addOutputTensor(o)
            outputs.append(o)

        return outputs

    def reference(ref_data):
        return [x[0] + np.transpose(x[1]) for x in zip(D0s, D1s)]

    # lowering the tolerance, as there are fp16 comparisons
    # We should find a way to lower the tolerance only
    # for the fp16 cases
    op_tester.rtol = 1e-02
    op_tester.atol = 1e-04
    op_tester.run(init_builder, reference, 'infer')
