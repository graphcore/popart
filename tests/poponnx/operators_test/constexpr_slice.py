# These slice tests are to verify the output of constexpr.
# To see that the const expr is actually being applied,
# see the tests in constexpr_tests directory
import numpy as np
from op_tester import op_tester


def test_slice_basic(op_tester):
    data = np.asarray([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32)
    dummy = np.asarray([[0, 0, 0]], dtype=np.float32)

    def init_builder(builder):
        c = builder.aiOnnx.constant(data)
        s = builder.aiOnnx.slice([c], axes=[0, 1], starts=[1, 0], ends=[2, 3])

        i1 = builder.addInputTensor(dummy)
        o = builder.aiOnnx.add([i1, s])
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
        c = builder.aiOnnx.constant(data)
        s = builder.aiOnnx.slice([c],
                                 axes=[1, 3, 4],
                                 starts=[1, 0, 2],
                                 ends=[3, 4, 4])

        i1 = builder.addInputTensor(dummy)
        o = builder.aiOnnx.add([i1, s])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        s = data[:, 1:3, :, 0:4, 2:4, :]
        result = dummy + s
        return [result]

    op_tester.passes = ['PreUniRepl']
    op_tester.run(init_builder, reference, 'infer')


def test_concat_axis_0(op_tester):
    _test_concat(op_tester, (2, 3, 4), 0)


def test_concat_axis_1(op_tester):
    _test_concat(op_tester, (2, 3, 4), 1)


def test_concat_axis_2(op_tester):
    _test_concat(op_tester, (2, 3, 4), 2)


def _test_concat(op_tester, shape, axis):
    dl = 1
    for i in shape:
        dl *= i

    d0 = np.arange(0, dl).reshape(shape).astype(np.float32)
    d1 = np.arange(dl + 1, 2 * dl + 1).reshape(shape).astype(np.float32)

    dummy = np.zeros(
        np.concatenate((d0, d1), axis=axis).shape, dtype=np.float32)

    def init_builder(builder):
        c0 = builder.aiOnnx.constant(d0)
        c1 = builder.aiOnnx.constant(d1)
        cc = builder.aiOnnx.concat([c0, c1], axis)

        i = builder.addInputTensor(dummy)
        o = builder.aiOnnx.add([i, cc])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        result = np.concatenate((d0, d1), axis=axis)
        return [result]

    op_tester.run(init_builder, reference, 'infer')


def test_concat_3_inputs(op_tester):
    shape = (2, 3, 4)
    dl = 2 * 3 * 4
    axis = 1

    d0 = np.arange(0, dl).reshape(shape).astype(np.float32)
    d1 = np.arange(dl + 1, 2 * dl + 1).reshape(shape).astype(np.float32)
    d2 = np.arange(2 * dl + 2, 3 * dl + 2).reshape(shape).astype(np.float32)

    dummy = np.zeros(
        np.concatenate((d0, d1, d2), axis=axis).shape, dtype=np.float32)

    def init_builder(builder):
        c0 = builder.aiOnnx.constant(d0)
        c1 = builder.aiOnnx.constant(d1)
        c2 = builder.aiOnnx.constant(d2)
        cc = builder.aiOnnx.concat([c0, c1, c2], axis)

        i = builder.addInputTensor(dummy)
        o = builder.aiOnnx.add([i, cc])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        result = np.concatenate((d0, d1, d2), axis=axis)
        return [result]

    op_tester.run(init_builder, reference, 'infer')
