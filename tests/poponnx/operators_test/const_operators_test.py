# These tests are to verify the output of constexpr.
# To see that the const expr is actually being applied,
# see the tests in `constexpr_test.cpp'
import numpy as np
from op_tester import op_tester


def test_slice_basic(op_tester):
    data = np.asarray([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32)
    dummy = np.asarray([[0, 0, 0]], dtype=np.float32)

    def init_builder(builder):
        c = builder.constant(data)
        s = builder.slice([c], [0, 1], [1, 0], [2, 3])

        i1 = builder.addInputTensor(dummy)
        o = builder.add([i1, s])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        result = np.asarray([[5, 6, 7]], dtype=np.float32)
        return [result]

    op_tester.passes = ['PreUniRepl']
    op_tester.run(init_builder, reference, 'infer')


def test_slice_complex(op_tester):
    data = np.random.rand(2, 3, 4, 5, 6, 7).astype(np.float32)
    dummy = np.zeros((2, 2, 4, 4, 2, 7), dtype=np.float32)

    def init_builder(builder):
        c = builder.constant(data)
        s = builder.slice([c], [1, 3, 4], [1, 0, 2], [3, 4, 4])

        i1 = builder.addInputTensor(dummy)
        o = builder.add([i1, s])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        s = data[:, 1:3, :, 0:4, 2:4, :]
        result = dummy + s
        return [result]

    op_tester.passes = ['PreUniRepl']
    op_tester.run(init_builder, reference, 'infer')
