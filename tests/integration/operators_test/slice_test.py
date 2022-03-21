# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
import torch
import pytest


# Slice1 used by opsets < 10 and aiGraphcore
@pytest.mark.parametrize("graphcore", (True, False))
def test_slice_opset1(op_tester, graphcore):
    d1 = np.array([[1., 2., 3., 4.], [5., 6., 7., 8.]]).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)

        if graphcore:
            o = builder.aiGraphcore.slice([i1],
                                          axes=[0, 1],
                                          starts=[1, 0],
                                          ends=[2, 3])
        else:
            o = builder.aiOnnxOpset9.slice([i1],
                                           axes=[0, 1],
                                           starts=[1, 0],
                                           ends=[2, 3])

        assert builder.getTensorShape(o) == [1, 3]
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        o = d1[1:2, 0:3]

        return [o]

    op_tester.run(init_builder, reference, 'infer')


def test_slice_opset10(op_tester):
    d1 = np.array([[1., 2., 3., 4.], [5., 6., 7., 8.]]).astype(np.float32)
    axesV = np.array([0, 1]).astype(np.int32)
    startsV = np.array([1, 0]).astype(np.int32)
    endsV = np.array([2, 3]).astype(np.int32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        axes = builder.addInitializedInputTensor(axesV)
        starts = builder.addInitializedInputTensor(startsV)
        ends = builder.addInitializedInputTensor(endsV)

        o = builder.aiOnnxOpset10.slice([i1, starts, ends, axes])
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        o = d1[1:2, 0:3]

        return [o]

    op_tester.run(init_builder, reference, 'infer')


def test_slice_default_axes(op_tester):
    d1 = np.array([[1., 2., 3., 4.], [5., 6., 7., 8.]]).astype(np.float32)
    startsV = np.array([1, 0]).astype(np.int32)
    endsV = np.array([2, 3]).astype(np.int32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        starts = builder.addInitializedInputTensor(startsV)
        ends = builder.addInitializedInputTensor(endsV)

        o = builder.aiOnnx.slice([i1, starts, ends])
        assert builder.getTensorShape(o) == [1, 3]
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        o = d1[1:2, 0:3]

        return [o]

    op_tester.run(init_builder, reference, 'infer')


@pytest.mark.parametrize("graphcore", (True, False))
def test_slice_neg(op_tester, graphcore):
    d1 = np.array([1., 2., 3., 4., 5., 6., 7., 8.]).astype(np.float32)
    axesV = np.array([0]).astype(np.int32)
    startsV = np.array([-5]).astype(np.int32)
    endsV = np.array([-3]).astype(np.int32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        axes = builder.addInitializedInputTensor(axesV)
        starts = builder.addInitializedInputTensor(startsV)
        ends = builder.addInitializedInputTensor(endsV)

        if graphcore:
            o = builder.aiGraphcore.slice([i1],
                                          starts=[startsV[0]],
                                          ends=[endsV[0]],
                                          axes=[axesV[0]])
        else:
            o = builder.aiOnnx.slice([i1, starts, ends, axes])

        assert builder.getTensorShape(o) == [2]

        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        o = d1[-5:-3]

        return [o]

    op_tester.run(init_builder, reference, 'infer')


@pytest.mark.parametrize("graphcore", (True, False))
def test_slice_grad(op_tester, graphcore):
    d1 = np.array([[1., 2., 3., 4.], [5., 6., 7., 8.]]).astype(np.float32)
    axesV = np.array([0, 1]).astype(np.int32)
    startsV = np.array([1, 0]).astype(np.int32)
    endsV = np.array([2, 3]).astype(np.int32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        axes = builder.aiOnnx.constant(axesV)
        starts = builder.aiOnnx.constant(startsV)
        ends = builder.aiOnnx.constant(endsV)

        if graphcore:
            o = builder.aiGraphcore.slice([i1],
                                          starts=[startsV[0], startsV[1]],
                                          ends=[endsV[0], endsV[1]],
                                          axes=[axesV[0], axesV[1]])
        else:
            o = builder.aiOnnx.slice([i1, starts, ends, axes])

        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + i1,
            popart.reservedGradientPrefix() + o
        ]

    def reference(ref_data):
        a = torch.tensor(d1, requires_grad=True)
        o = a[1:2, 0:3]

        d__o = ref_data.getOutputTensorGrad(0)

        o.backward(torch.tensor(d__o))

        return [o, a.grad, None]

    op_tester.setPatterns(['PreUniRepl'], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'train')


def test_slice_error_start_input(op_tester):
    d1 = np.array([[1., 2., 3., 4.], [5., 6., 7., 8.]]).astype(np.float32)
    axesV = np.array([0, 1]).astype(np.int32)
    startsV = np.array([1, 0]).astype(np.int32)
    endsV = np.array([2, 3]).astype(np.int32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        starts = builder.addInputTensor(startsV)
        ends = builder.addInputTensor(endsV)

        o = builder.aiOnnx.slice([i1, starts, ends])
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        return []

    op_tester.setPatterns(['PreUniRepl'], enableRuntimeAsserts=False)
    with pytest.raises(popart.popart_exception) as e_info:
        op_tester.run(init_builder, reference, 'train')

    assert (
        e_info.value.args[0] ==
        "Need the value of the ai.onnx.Slice:10 input 'starts' to determine the "
        "output shape, but was unable because "
        "[Tensor::getDataViaGraphTraversal] Could not work out tensor data for "
        "input/1.")


@pytest.mark.parametrize("graphcore", (True, False))
def test_slice_start_out_of_bounds(op_tester, graphcore):
    """
    The slice bounds tests follow the behaviour asserted by the Onnx tests,
    which follow the behaviour of numpy.
    https://github.com/onnx/onnx/blob/master/onnx/backend/test/case/node/slice.py

    For a dimension of size n, any slice index of m > n, becomes n. That is, a
    slice 10:21 on a dimension of size 20, becomes 10:20.

    Note further that an a:b slice is the open-closed interval [a, b), so in the
    above example, a slice of 10:20 is valid.

    A slice of 20:20, though, is also valid in numpy; it becomes a dimension of
    size 0 (but the other dimensions are not affected). The array will have zero
    elements.
    """

    d1 = np.random.randn(20, 10, 5).astype(np.float32)

    # Will create a zero-dim slice, as 1000:1000 becomes 10:10.
    axesV = np.array([1], dtype=np.int64)
    startsV = np.array([1000], np.int64)
    endsV = np.array([1000], np.int64)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        axes = builder.aiOnnx.constant(axesV)
        starts = builder.aiOnnx.constant(startsV)
        ends = builder.aiOnnx.constant(endsV)

        if graphcore:
            o = builder.aiGraphcore.slice([i1],
                                          starts=[startsV[0]],
                                          ends=[endsV[0]],
                                          axes=[axesV[0]])
        else:
            o = builder.aiOnnx.slice([i1, starts, ends, axes])
        assert builder.getTensorShape(o) == [20, 0, 5]

        builder.addOutputTensor(o)

        return [o, popart.reservedGradientPrefix() + i1]

    def reference(_):  # ref_data is an unused argument
        o = d1[:, 1000:1000]
        i1_grad = np.zeros(d1.shape, dtype=np.float32)

        return [o, i1_grad]

    op_tester.run(init_builder, reference, 'train')


@pytest.mark.parametrize("graphcore", (True, False))
def test_slice_end_out_of_bounds(op_tester, graphcore):
    """
    The slice bounds tests follow the behaviour asserted by the Onnx tests,
    which follow the behaviour of numpy.
    https://github.com/onnx/onnx/blob/master/onnx/backend/test/case/node/slice.py

    For a dimension of size n, any slice index of m > n, becomes n. That is, a
    slice 10:21 on a dimension of size 20, becomes 10:20.

    Note further that an a:b slice is the open-closed interval [a, b), so in the
    above example, a slice of 10:20 is valid.

    A slice of 20:20, though, is also valid in numpy; it becomes a dimension of
    size 0 (but the other dimensions are not affected). The array will have zero
    elements.
    """

    d1 = np.random.randn(20, 10, 5).astype(np.float32)

    # Will create a (20, 9, 5)-dim slice, as 1:1000 becomes 1:10.
    axesV = np.array([1], dtype=np.int64)
    startsV = np.array([1], dtype=np.int64)
    endsV = np.array([1000], dtype=np.int64)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        axes = builder.aiOnnx.constant(axesV)
        starts = builder.aiOnnx.constant(startsV)
        ends = builder.aiOnnx.constant(endsV)

        if graphcore:
            o = builder.aiGraphcore.slice([i1],
                                          starts=[startsV[0]],
                                          ends=[endsV[0]],
                                          axes=[axesV[0]])
        else:
            o = builder.aiOnnx.slice([i1, starts, ends, axes])

        assert builder.getTensorShape(o) == [20, 9, 5]
        builder.addOutputTensor(o)

        return [
            o,
            popart.reservedGradientPrefix() + i1,
            popart.reservedGradientPrefix() + o
        ]

    def reference(ref_data):
        o = d1[:, 1:1000]

        o_grad = np.ones(o.shape,
                         dtype=np.float32) * ref_data.getOutputTensorGrad(0)
        i1_grad = np.pad(o_grad, [(0, 0), (1, 0), (0, 0)], constant_values=0.)

        return [o, i1_grad, None]

    op_tester.run(init_builder, reference, 'train')


@pytest.mark.parametrize("graphcore", (True, False))
def test_slice_neg_starts_and_ends(op_tester, graphcore):
    d1 = np.array([1., 2., 3., 4.]).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        if graphcore:
            o = builder.aiGraphcore.slice([i1],
                                          axes=[0],
                                          starts=[-5],
                                          ends=[-1])
            assert builder.getTensorShape(o) == [3]
        else:
            o = builder.aiOnnxOpset9.slice([i1],
                                           axes=[0],
                                           starts=[-5],
                                           ends=[-1])
        builder.addOutputTensor(o)

        return [o]

    def reference(_):  # ref_data is an unused argument
        o = d1[-4:-1]

        return [o]

    op_tester.run(init_builder, reference, 'infer')


def test_slice_flip_1(op_tester):
    d1 = np.array([1., 2., 3., 4.]).astype(np.float32)
    axesV = np.array([0], dtype=np.int64)
    startsV = np.array([3], dtype=np.int64)
    endsV = np.array([1], dtype=np.int64)
    stepsV = np.array([-1], dtype=np.int64)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        axes = builder.aiOnnx.constant(axesV)
        starts = builder.aiOnnx.constant(startsV)
        ends = builder.aiOnnx.constant(endsV)
        steps = builder.aiOnnx.constant(stepsV)

        o = builder.aiOnnx.slice([i1, starts, ends, axes, steps])

        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        o = d1[3:1:-1]

        return [o]

    op_tester.run(init_builder, reference, 'infer')


def test_slice_flip_2(op_tester):
    d1 = np.array([1., 2., 3., 4.]).astype(np.float32)
    axesV = np.array([0], dtype=np.int64)
    startsV = np.array([-1], dtype=np.int64)
    endsV = np.array([-1000], dtype=np.int64)
    stepsV = np.array([-1], dtype=np.int64)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        axes = builder.aiOnnx.constant(axesV)
        starts = builder.aiOnnx.constant(startsV)
        ends = builder.aiOnnx.constant(endsV)
        steps = builder.aiOnnx.constant(stepsV)

        o = builder.aiOnnx.slice([i1, starts, ends, axes, steps])

        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        return [np.flip(d1)]

    op_tester.run(init_builder, reference, 'infer')


def test_slice_flip_grad_1(op_tester):
    d1 = np.array([1., 2., 3., 4., 5.]).astype(np.float32)
    axesV = np.array([0], dtype=np.int64)

    starts0V = np.array([4], dtype=np.int64)
    ends0V = np.array([1], dtype=np.int64)
    stepsV = np.array([-1], dtype=np.int64)

    starts1V = np.array([1], dtype=np.int64)
    ends1V = np.array([3], dtype=np.int64)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        axes = builder.aiOnnx.constant(axesV)
        starts = builder.aiOnnx.constant(starts0V)
        ends = builder.aiOnnx.constant(ends0V)
        steps = builder.aiOnnx.constant(stepsV)

        o = builder.aiOnnx.slice([i1, starts, ends, axes, steps])

        starts = builder.aiOnnx.constant(starts1V)
        ends = builder.aiOnnx.constant(ends1V)

        o = builder.aiOnnx.slice([o, starts, ends, axes])

        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + i1,
            popart.reservedGradientPrefix() + o
        ]

    def reference(ref_data):
        a = torch.tensor(d1, requires_grad=True)
        o = torch.flip(a[2:5], [0])
        o = o[1:3]
        d__o = ref_data.getOutputTensorGrad(0)

        o.backward(torch.tensor(d__o))

        print(o)
        print(a.grad)
        return [o, a.grad, None]

    op_tester.setPatterns(['PreUniRepl'], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'train')
