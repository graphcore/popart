# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
# This is the second half of operators_test.py, split to avoid timeouts on debug builds.
# Required to avoid "tensor is not callable pylint error:"
# pylint: disable=E1102
import numpy as np
import pytest
import torch

import popart


def test_argmin_no_keepdims(op_tester):
    d1 = np.random.rand(5, 7, 11, 13).astype(np.float32)
    axis = 0
    keepdims = 0

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.argmin([i1], axis, keepdims, "test_argmin")
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        result = np.argmin(d1, axis=axis)
        return [result.astype(np.int32)]

    op_tester.run(init_builder, reference, "infer")


def test_argmin_keepdims(op_tester):
    d1 = np.random.rand(5, 7, 11, 13).astype(np.float32)
    axis = 0
    keepdims = 1

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.argmin([i1], axis, keepdims, "test_argmin")
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        result = np.argmin(d1, axis=axis)
        result = np.expand_dims(result, axis)
        return [result.astype(np.int32)]

    op_tester.run(init_builder, reference, "infer")


def test_argmin_negative_axis(op_tester):
    d1 = np.random.rand(5, 7, 11, 13).astype(np.float32)
    axis = -1
    keepdims = 1

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.argmin([i1], axis, keepdims, "test_argmin")
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        result = np.argmin(d1, axis=axis)
        result = np.expand_dims(result, axis)
        return [result.astype(np.int32)]

    op_tester.run(
        init_builder, reference, "infer", opsets={"ai.onnx": 11, "ai.graphcore": 1}
    )


def _test_argmax(op_tester, data, axis, keepdims, opsets):
    def init_builder(builder):
        i1 = builder.addInputTensor(data)
        o = builder.aiOnnx.argmax([i1], axis, keepdims, "test_argmax")
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        result = np.argmax(data, axis=axis)
        if keepdims == 1:
            result = np.expand_dims(result, axis)
        return [result.astype(np.int32)]

    op_tester.run(init_builder, reference, "infer", opsets=opsets)


def test_argmax_2d(op_tester):
    data = np.random.rand(5, 6).astype(np.float32)
    opsets = {"ai.onnx": 9, "ai.graphcore": 1}
    _test_argmax(op_tester, data, 0, 1, opsets)
    _test_argmax(op_tester, data, 0, 0, opsets)
    _test_argmax(op_tester, data, 1, 1, opsets)
    _test_argmax(op_tester, data, 1, 0, opsets)

    # Test negative axis index for onnx opset 11
    opsets = {"ai.onnx": 11, "ai.graphcore": 1}
    _test_argmax(op_tester, data, -1, 0, opsets)
    _test_argmax(op_tester, data, -2, 0, opsets)


def test_argmax_no_keepdims(op_tester):
    d1 = np.random.rand(5, 7, 11, 13).astype(np.float32)
    axis = 0
    keepdims = 0

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.argmax([i1], axis, keepdims, "test_argmax")
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        result = np.argmax(d1, axis=axis)
        return [result.astype(np.int32)]

    op_tester.run(init_builder, reference, "infer")


def test_ceil(op_tester):
    d1 = np.random.rand(2, 7).astype(np.float32)
    d1 = d1 * 6 - 3  # numbers in range [-3, 3]

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.ceil([i1], "test_ceil")
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        result = np.ceil(d1)
        return [result.astype(np.float32)]

    op_tester.run(init_builder, reference, "infer")


def test_ceil_inplace(op_tester):
    d1 = np.random.rand(2, 7).astype(np.float32)
    d1 = d1 * 10  # numbers in range [0, 10]

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        # Pad with ops to allow in-placing
        log = builder.aiOnnx.log([i1])
        ceil = builder.aiOnnx.ceil([log], "test_ceil")
        o = builder.aiOnnx.exp([ceil])
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        result = np.exp(np.ceil(np.log(d1)))
        return [result.astype(np.float32)]

    op_tester.setPatterns(["InPlace"], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, "infer")


def test_ceil_grad(op_tester):
    d1 = np.random.rand(2, 7).astype(np.float32)
    d1 = d1 * 6 - 3  # numbers in range [-3, 3]

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.ceil([i1], "test_ceil")
        builder.addOutputTensor(o)
        return [o, popart.reservedGradientPrefix() + i1]

    def reference(_):  # ref_data is an unused argument
        return [np.ceil(d1).astype(np.float32), np.zeros_like(d1)]

    op_tester.run(init_builder, reference, "train")


def test_floor(op_tester):
    d1 = np.random.rand(2, 7).astype(np.float32)
    d1 = d1 * 6 - 3  # numbers in range [-3, 3]

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.floor([i1], "test_floor")
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        result = np.floor(d1)
        return [result.astype(np.float32)]

    op_tester.run(init_builder, reference, "infer")


def test_floor_inplace(op_tester):
    d1 = np.random.rand(2, 7).astype(np.float32)
    d1 = d1 * 10  # numbers in range [0, 10]

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        # Pad with ops to allow in-placing
        log = builder.aiOnnx.log([i1])
        floor = builder.aiOnnx.floor([log], "test_floor")
        o = builder.aiOnnx.exp([floor])
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        result = np.exp(np.floor(np.log(d1)))
        return [result.astype(np.float32)]

    op_tester.setPatterns(["InPlace"], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, "infer")


def test_floor_grad(op_tester):
    d1 = np.random.rand(2, 7).astype(np.float32)
    d1 = d1 * 6 - 3  # numbers in range [-3, 3]

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.floor([i1], "test_floor")
        builder.addOutputTensor(o)
        return [o, popart.reservedGradientPrefix() + i1]

    def reference(_):  # ref_data is an unused argument
        return [np.floor(d1).astype(np.float32), np.zeros_like(d1)]

    op_tester.run(init_builder, reference, "train")


def test_clip(op_tester):
    d1 = np.random.rand(2, 7).astype(np.float32)
    d1 = d1 * 6 - 3  # numbers in range [-3, 3]

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.clip([i1], min=-1.5, max=1.5)
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        a = torch.tensor(d1)
        result = torch.clamp(a, min=-1.5, max=1.5)
        return [result]

    op_tester.run(init_builder, reference, "infer")


def test_clip_inplace(op_tester):
    d1 = np.random.rand(2, 7).astype(np.float32)
    d1 = d1 * 10  # numbers in range [0, 10]

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        # Pad with ops to allow in-placing
        log = builder.aiOnnx.log([i1])
        clip = builder.aiOnnx.clip([log], min=4, max=7)
        o = builder.aiOnnx.exp([clip])
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        a = torch.tensor(d1)
        result = torch.exp(torch.clamp(torch.log(a), min=4, max=7))
        return [result]

    op_tester.setPatterns(["InPlace"], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, "infer")


def test_clip_grad(op_tester):
    d1 = np.random.rand(2, 7).astype(np.float32)
    d1 = d1 * 6 - 3  # numbers in range [-3, 3]

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.clip([i1], min=-1.5, max=1.5)
        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + i1,
            popart.reservedGradientPrefix() + o,
        ]

    def reference(ref_data):
        a = torch.tensor(d1, requires_grad=True)
        b = torch.clamp(a, min=-1.5, max=1.5)
        d__o = ref_data.getOutputTensorGrad(0)
        b.backward(torch.tensor(d__o))
        print(b)
        print(a.grad)
        print("b grad", b.grad)
        return [b, a.grad, None]

    op_tester.setPatterns(["PreUniRepl"], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, "train")


def test_clip11(op_tester):
    d1 = np.random.rand(2, 7).astype(np.float32)
    d1 = d1 * 6 - 3  # numbers in range [-3, 3]
    d_min = np.array([-1.5], dtype=np.float32)
    d_max = np.array([1.5], dtype=np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        t_min = builder.aiOnnx.constant(d_min, False)
        t_max = builder.aiOnnx.constant(d_max, False)
        o = builder.aiOnnx.clip([i1, t_min, t_max])
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        a = torch.tensor(d1)
        result = torch.clamp(a, min=d_min[0], max=d_max[0])
        return [result]

    op_tester.run(
        init_builder, reference, "infer", opsets={"ai.onnx": 11, "ai.graphcore": 1}
    )


def test_clip11_default_min(op_tester):
    d1 = np.random.rand(2, 7).astype(np.float32)
    d1 = d1 * 6 - 3  # numbers in range [-3, 3]
    d_max = np.array([1.5], dtype=np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        t_max = builder.aiOnnx.constant(d_max, False)
        o = builder.aiOnnx.clip([i1, "", t_max])
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        a = torch.tensor(d1)
        result = torch.clamp(a, min=torch.finfo(torch.float32).min, max=d_max[0])
        return [result]

    op_tester.run(
        init_builder, reference, "infer", opsets={"ai.onnx": 11, "ai.graphcore": 1}
    )


def test_clip11_default_max(op_tester):
    d1 = np.random.rand(2, 7).astype(np.float32)
    d1 = d1 * 6 - 3  # numbers in range [-3, 3]
    d_min = np.array([-1.5], dtype=np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        t_min = builder.aiOnnx.constant(d_min, False)
        o = builder.aiOnnx.clip([i1, t_min])
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        a = torch.tensor(d1)
        # result = torch.clamp(a, min=d_min[0], max=torch.finfo(torch.float32).max)
        result = torch.clamp(a, min=d_min[0], max=100)
        return [result]

    op_tester.run(
        init_builder, reference, "infer", opsets={"ai.onnx": 11, "ai.graphcore": 1}
    )


def test_argmax_keepdims(op_tester):
    d1 = np.random.rand(5, 7, 11, 13).astype(np.float32)
    axis = 0
    keepdims = 1

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.argmax([i1], axis, keepdims, "test_argmax")
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        result = np.argmax(d1, axis=axis)
        result = np.expand_dims(result, axis)
        return [result.astype(np.int32)]

    op_tester.run(init_builder, reference, "infer")


def test_instancenorm_grad(op_tester):
    batch_size = 3
    features = 3
    width = 4
    height = 4

    non_zero_places = 5

    data = np.random.rand(batch_size, features, width, height).astype(np.float32)

    a_few_places = np.zeros_like(data.flatten())

    for _ in range(non_zero_places):
        a_few_places[int(np.random.rand() * a_few_places.size)] = 1.0

    a_few_places = a_few_places.reshape(*data.shape)

    scale = np.random.rand(features).astype(np.float32)
    bias = np.random.rand(features).astype(np.float32)

    epsilon = 1e-05

    def init_builder(builder):

        i_data = builder.addInputTensor(data)
        i_scale = builder.addInputTensor(scale)
        i_bias = builder.addInputTensor(bias)

        i_few_places = builder.addInputTensor(a_few_places)

        normed = builder.aiOnnx.instancenormalization(
            [i_data, i_scale, i_bias], epsilon
        )
        out = builder.aiOnnx.mul([normed, i_few_places])

        builder.addOutputTensor(out)

        return [
            out,
            normed,
            popart.reservedGradientPrefix() + i_data,
            popart.reservedGradientPrefix() + i_scale,
            popart.reservedGradientPrefix() + i_bias,
            popart.reservedGradientPrefix() + out,
        ]

    def reference(ref_data):
        i_data = torch.tensor(data, requires_grad=True)

        m = torch.nn.InstanceNorm2d(features, eps=epsilon, momentum=0, affine=True)
        m.weight.data = torch.tensor(scale)
        m.bias.data = torch.tensor(bias)
        normed = m(i_data)

        out = normed * torch.tensor(a_few_places, requires_grad=False)

        d__o = ref_data.getOutputTensorGrad(0)

        out.backward(torch.tensor(d__o))

        assert i_data.grad is not None
        assert m.weight.grad is not None
        assert m.bias.grad is not None

        return [out, normed, i_data.grad, m.weight.grad, m.bias.grad, None]

    # We decrease atol as the reference tensor computed has slightly changed in
    # torch 1.10 and the test fails on one value very close to 0.
    op_tester.atol = 1e-6

    op_tester.setPatterns(
        ["PreUniRepl", "ReciprocalGradOp", "MulArgGradOp"], enableRuntimeAsserts=False
    )
    op_tester.run(init_builder, reference, "train")


def test_instancenorm_grad_5D_input(op_tester):
    batch_size = 3
    features = 3
    d1 = 4
    d2 = 4
    d3 = 4

    non_zero_places = 10

    data = np.random.rand(batch_size, features, d1, d2, d3).astype(np.float32)

    a_few_places = np.zeros_like(data.flatten())

    for _ in range(non_zero_places):
        a_few_places[int(np.random.rand() * a_few_places.size)] = 1.0

    a_few_places = a_few_places.reshape(*data.shape)

    scale = np.random.rand(features).astype(np.float32)
    bias = np.random.rand(features).astype(np.float32)

    epsilon = 1e-05

    def init_builder(builder):

        i_data = builder.addInputTensor(data)
        i_scale = builder.addInputTensor(scale)
        i_bias = builder.addInputTensor(bias)

        i_few_places = builder.addInputTensor(a_few_places)

        normed = builder.aiOnnx.instancenormalization(
            [i_data, i_scale, i_bias], epsilon
        )
        out = builder.aiOnnx.mul([normed, i_few_places])

        builder.addOutputTensor(out)

        return [
            out,
            normed,
            popart.reservedGradientPrefix() + i_data,
            popart.reservedGradientPrefix() + i_scale,
            popart.reservedGradientPrefix() + i_bias,
            popart.reservedGradientPrefix() + out,
        ]

    def reference(ref_data):
        i_data = torch.tensor(data, requires_grad=True)

        m = torch.nn.InstanceNorm3d(features, eps=epsilon, momentum=0, affine=True)
        m.weight.data = torch.tensor(scale)
        m.bias.data = torch.tensor(bias)
        normed = m(i_data)

        out = normed * torch.tensor(a_few_places, requires_grad=False)

        d__o = ref_data.getOutputTensorGrad(0)

        out.backward(torch.tensor(d__o))

        assert i_data.grad is not None
        assert m.weight.grad is not None
        assert m.bias.grad is not None

        return [out, normed, i_data.grad, m.weight.grad, m.bias.grad, None]

    op_tester.atol *= 10
    op_tester.setPatterns(
        ["PreUniRepl", "ReciprocalGradOp", "MulArgGradOp"], enableRuntimeAsserts=False
    )
    op_tester.run(init_builder, reference, "train")


def test_constantofshape(op_tester):
    shape = np.array([1, 2, 3]).astype(np.int64)
    value = np.array([3.1415]).astype(np.float32)

    def init_builder(builder):
        i = builder.aiOnnx.constant(shape)
        c = builder.aiOnnx.constantofshape([i], value)
        o = builder.aiOnnx.identity([c])

        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        out = np.array([3.1415] * 2 * 3).astype(np.float32)
        out = np.reshape(out, (1, 2, 3))
        return [out]

    op_tester.run(init_builder, reference, "infer")


def test_concat(op_tester):
    values = [np.random.rand(1, 2, 3).astype(np.float32) for i in range(4)]

    def init_builder(builder):
        i = [builder.addInputTensor(v) for v in values]
        c = builder.aiOnnx.concat(i, 1)
        o = builder.aiOnnx.identity([c])

        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        out = np.concatenate(values, 1)
        return [out]

    op_tester.run(init_builder, reference, "infer")


def test_concat_negative_axis(op_tester):
    values = [np.random.rand(1, 2, 2, 2).astype(np.float32) for i in range(4)]

    def init_builder(builder):
        i = [builder.addInputTensor(v) for v in values]
        c = builder.aiOnnx.concat(i, -1)
        o = builder.aiOnnx.identity([c])

        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        out = np.concatenate(values, -1)
        return [out]

    op_tester.run(init_builder, reference, "infer")


def test_constant_of_shape(op_tester):
    data = np.random.rand(2, 2).astype(np.float32)
    shape_data = np.array([2, 2]).astype(np.int64)

    def init_builder(builder):
        in0 = builder.addInputTensor(data)
        c0 = builder.aiOnnx.constant(shape_data)
        c = builder.aiOnnx.constantofshape([c0], np.array([5], dtype=np.float32))
        o = builder.aiOnnx.add([in0, c])

        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        o = data + np.array([5], dtype=np.float32)
        return [o]

    op_tester.run(init_builder, reference, "infer")


def test_constant_of_shape_int32(op_tester):
    data = np.random.rand(2, 2).astype(np.float32)
    shape_data = np.array([2, 2]).astype(np.int32)

    def init_builder(builder):
        in0 = builder.addInputTensor(data)
        c0 = builder.aiOnnx.constant(shape_data)
        c = builder.aiOnnx.constantofshape([c0], np.array([5], dtype=np.float32))
        o = builder.aiOnnx.add([in0, c])

        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        o = data + np.array([5], dtype=np.float32)
        return [o]

    op_tester.run(init_builder, reference, "infer")


def test_convtranspose(op_tester):
    # Test modified from example `convtranspose` in onnx
    # operators documentation.
    x = np.array(
        [[[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]]]  # (1, 1, 3, 3)
    ).astype(np.float32)

    W = np.array(
        [
            [
                [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],  # (1, 2, 3, 3)
                [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
            ]
        ]
    ).astype(np.float32)

    def init_builder(builder):
        d = builder.addInputTensor(x)
        f = builder.addInputTensor(W)
        o = builder.aiOnnxOpset11.convtranspose([d, f])
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        y = np.array(
            [
                [
                    [
                        [0.0, 1.0, 3.0, 3.0, 2.0],  # (1, 2, 5, 5)
                        [3.0, 8.0, 15.0, 12.0, 7.0],
                        [9.0, 21.0, 36.0, 27.0, 15.0],
                        [9.0, 20.0, 33.0, 24.0, 13.0],
                        [6.0, 13.0, 21.0, 15.0, 8.0],
                    ],
                    [
                        [0.0, 1.0, 3.0, 3.0, 2.0],
                        [3.0, 8.0, 15.0, 12.0, 7.0],
                        [9.0, 21.0, 36.0, 27.0, 15.0],
                        [9.0, 20.0, 33.0, 24.0, 13.0],
                        [6.0, 13.0, 21.0, 15.0, 8.0],
                    ],
                ]
            ]
        ).astype(np.float32)
        return [y]

    op_tester.setPatterns(["ConvTranspose"], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, step_type="infer")


@pytest.mark.parametrize("with_pattern", [True, False])
def test_convtranspose_grad(op_tester, with_pattern):
    x = np.array(
        [
            [
                [
                    [-0.6548, 2.1528, -0.7060, 1.2557],
                    [0.5779, 1.6015, 0.0200, -0.5367],
                    [0.9002, -1.0468, -0.5903, -0.9868],
                    [-0.0274, 0.0813, 0.3434, 0.0679],
                ]
            ]
        ]
    ).astype(np.float32)
    W = np.array([[[[-0.2817, 0.2606], [-0.4299, -0.4982]]]]).astype(np.float32)

    def init_builder(builder):
        d = builder.addInputTensor(x)
        f = builder.addInitializedInputTensor(W)
        o = builder.aiOnnxOpset11.convtranspose([d, f])
        two = builder.aiOnnxOpset11.constant(
            np.array([2.0], dtype=np.float32), False, "two"
        )
        o_sq = builder.aiOnnxOpset11.pow([o, two])

        builder.addOutputTensor(o_sq)
        return [
            o_sq,
            popart.reservedGradientPrefix() + d,
            popart.reservedGradientPrefix() + f,
            popart.reservedGradientPrefix() + o_sq,
        ]

    def reference(ref_data):
        t1 = torch.tensor(x, requires_grad=True)
        t2 = torch.tensor(W, requires_grad=True)

        out = torch.nn.functional.conv_transpose2d(t1, t2)

        # Ensure the gradient varies throughout with a non-linearity
        out = out ** 2.0
        print(out)

        d__o = torch.tensor(ref_data.getOutputTensorGrad(0))
        out.backward(d__o)

        print(t1.grad)

        return [out, t1.grad, t2.grad, None]

    patterns = [
        "ConvTranspose",
        "ConvFlipWeightsGradOp",
        "ConvDataGrad",
        "PowArg0GradOp",
        "SqrtGradOp",
    ]

    if with_pattern:
        patterns.append("ConvFlipWeightsDoubleFlip")

    op_tester.setPatterns(patterns, enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, step_type="train")


def test_convtranspose_auto_pad(op_tester):
    # Test when the output shape is specified and that the padding is automatically
    # computed.
    x = np.array([[[[1.0, 2.0], [3.0, 4.0]]]]).astype(np.float32)  # (1, 1, 2, 2)

    W = np.array([[[[1.0, 2.0], [2.0, 1.0]]]]).astype(np.float32)  # (1, 1, 2, 2)

    def init_builder(builder):
        d = builder.addInputTensor(x)
        f = builder.addInputTensor(W)
        o = builder.aiOnnxOpset11.convtranspose([d, f], output_shape=[2, 2])
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        y = np.array([[[[1.0, 4.0], [5.0, 15.0]]]]).astype(np.float32)  # (1, 1, 2, 2)
        return [y]

    op_tester.setPatterns(["ConvTranspose"], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, step_type="infer")


def test_convtranspose_auto_pad_same_upper(op_tester):
    # Test when the output shape is specified and that the padding is automatically
    # computed and that auto_pad is set to SAME_UPPER
    x = np.array([[[[1.0, 2.0], [3.0, 4.0]]]]).astype(np.float32)  # (1, 1, 2, 2)

    W = np.array([[[[1.0, 2.0], [2.0, 1.0]]]]).astype(np.float32)  # (1, 1, 2, 2)

    def init_builder(builder):
        d = builder.addInputTensor(x)
        f = builder.addInputTensor(W)
        o = builder.aiOnnxOpset11.convtranspose([d, f], output_shape=[2, 2])
        # Work-around to set auto_pad until T38874 is done:
        builder.addNodeAttribute("auto_pad", "SAME_UPPER", set([o]))
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        y = np.array([[[[15.0, 10.0], [11.0, 4.0]]]]).astype(np.float32)  # (1, 1, 2, 2)
        return [y]

    op_tester.setPatterns(["ConvTranspose"], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, step_type="infer")


def test_convtranspose_1d(op_tester):
    # Test modified from example `convtranspose_1d` in onnx
    # operators documentation.
    x = np.array([[[0.0, 1.0, 2.0]]]).astype(np.float32)  # (1, 1, 3)

    W = np.array([[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]).astype(np.float32)  # (1, 2, 3)

    def init_builder(builder):
        d = builder.addInputTensor(x)
        f = builder.addInputTensor(W)
        o = builder.aiOnnxOpset11.convtranspose([d, f])
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        y = np.array(
            [[[0.0, 1.0, 3.0, 3.0, 2.0], [0.0, 1.0, 3.0, 3.0, 2.0]]]  # (1, 2, 5)
        ).astype(np.float32)
        return [y]

    op_tester.setPatterns(["ConvTranspose"], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, step_type="infer")


def test_convtranspose_3d(op_tester):
    # Test modified from example `convtranspose_3d` in onnx
    # operators documentation.
    x = np.array(
        [
            [
                [
                    [
                        [0.0, 1.0, 2.0, 3.0, 4.0],
                        [5.0, 6.0, 7.0, 8.0, 9.0],
                        [10.0, 11.0, 12.0, 13.0, 14.0],
                        [15.0, 16.0, 17.0, 18.0, 19.0],
                    ],
                    [
                        [20.0, 21.0, 22.0, 23.0, 24.0],
                        [25.0, 26.0, 27.0, 28.0, 29.0],
                        [30.0, 31.0, 32.0, 33.0, 34.0],
                        [35.0, 36.0, 37.0, 38.0, 39.0],
                    ],
                    [
                        [40.0, 41.0, 42.0, 43.0, 44.0],
                        [45.0, 46.0, 47.0, 48.0, 49.0],
                        [50.0, 51.0, 52.0, 53.0, 54.0],
                        [55.0, 56.0, 57.0, 58.0, 59.0],
                    ],
                ]
            ]
        ]
    ).astype(np.float32)

    W = np.array(
        [
            [
                [
                    [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                    [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                    [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                ],
                [
                    [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                    [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                    [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                ],
            ]
        ]
    ).astype(np.float32)

    def init_builder(builder):
        d = builder.addInputTensor(x)
        f = builder.addInputTensor(W)
        o = builder.aiOnnxOpset11.convtranspose([d, f])
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        y = np.array(
            [
                [
                    [
                        [
                            [0.0, 1.0, 3.0, 6.0, 9.0, 7.0, 4.0],
                            [5.0, 12.0, 21.0, 27.0, 33.0, 24.0, 13.0],
                            [15.0, 33.0, 54.0, 63.0, 72.0, 51.0, 27.0],
                            [30.0, 63.0, 99.0, 108.0, 117.0, 81.0, 42.0],
                            [25.0, 52.0, 81.0, 87.0, 93.0, 64.0, 33.0],
                            [15.0, 31.0, 48.0, 51.0, 54.0, 37.0, 19.0],
                        ],
                        [
                            [20.0, 42.0, 66.0, 72.0, 78.0, 54.0, 28.0],
                            [50.0, 104.0, 162.0, 174.0, 186.0, 128.0, 66.0],
                            [90.0, 186.0, 288.0, 306.0, 324.0, 222.0, 114.0],
                            [120.0, 246.0, 378.0, 396.0, 414.0, 282.0, 144.0],
                            [90.0, 184.0, 282.0, 294.0, 306.0, 208.0, 106.0],
                            [50.0, 102.0, 156.0, 162.0, 168.0, 114.0, 58.0],
                        ],
                        [
                            [60.0, 123.0, 189.0, 198.0, 207.0, 141.0, 72.0],
                            [135.0, 276.0, 423.0, 441.0, 459.0, 312.0, 159.0],
                            [225.0, 459.0, 702.0, 729.0, 756.0, 513.0, 261.0],
                            [270.0, 549.0, 837.0, 864.0, 891.0, 603.0, 306.0],
                            [195.0, 396.0, 603.0, 621.0, 639.0, 432.0, 219.0],
                            [105.0, 213.0, 324.0, 333.0, 342.0, 231.0, 117.0],
                        ],
                        [
                            [60.0, 122.0, 186.0, 192.0, 198.0, 134.0, 68.0],
                            [130.0, 264.0, 402.0, 414.0, 426.0, 288.0, 146.0],
                            [210.0, 426.0, 648.0, 666.0, 684.0, 462.0, 234.0],
                            [240.0, 486.0, 738.0, 756.0, 774.0, 522.0, 264.0],
                            [170.0, 344.0, 522.0, 534.0, 546.0, 368.0, 186.0],
                            [90.0, 182.0, 276.0, 282.0, 288.0, 194.0, 98.0],
                        ],
                        [
                            [40.0, 81.0, 123.0, 126.0, 129.0, 87.0, 44.0],
                            [85.0, 172.0, 261.0, 267.0, 273.0, 184.0, 93.0],
                            [135.0, 273.0, 414.0, 423.0, 432.0, 291.0, 147.0],
                            [150.0, 303.0, 459.0, 468.0, 477.0, 321.0, 162.0],
                            [105.0, 212.0, 321.0, 327.0, 333.0, 224.0, 113.0],
                            [55.0, 111.0, 168.0, 171.0, 174.0, 117.0, 59.0],
                        ],
                    ],
                    [
                        [
                            [0.0, 1.0, 3.0, 6.0, 9.0, 7.0, 4.0],
                            [5.0, 12.0, 21.0, 27.0, 33.0, 24.0, 13.0],
                            [15.0, 33.0, 54.0, 63.0, 72.0, 51.0, 27.0],
                            [30.0, 63.0, 99.0, 108.0, 117.0, 81.0, 42.0],
                            [25.0, 52.0, 81.0, 87.0, 93.0, 64.0, 33.0],
                            [15.0, 31.0, 48.0, 51.0, 54.0, 37.0, 19.0],
                        ],
                        [
                            [20.0, 42.0, 66.0, 72.0, 78.0, 54.0, 28.0],
                            [50.0, 104.0, 162.0, 174.0, 186.0, 128.0, 66.0],
                            [90.0, 186.0, 288.0, 306.0, 324.0, 222.0, 114.0],
                            [120.0, 246.0, 378.0, 396.0, 414.0, 282.0, 144.0],
                            [90.0, 184.0, 282.0, 294.0, 306.0, 208.0, 106.0],
                            [50.0, 102.0, 156.0, 162.0, 168.0, 114.0, 58.0],
                        ],
                        [
                            [60.0, 123.0, 189.0, 198.0, 207.0, 141.0, 72.0],
                            [135.0, 276.0, 423.0, 441.0, 459.0, 312.0, 159.0],
                            [225.0, 459.0, 702.0, 729.0, 756.0, 513.0, 261.0],
                            [270.0, 549.0, 837.0, 864.0, 891.0, 603.0, 306.0],
                            [195.0, 396.0, 603.0, 621.0, 639.0, 432.0, 219.0],
                            [105.0, 213.0, 324.0, 333.0, 342.0, 231.0, 117.0],
                        ],
                        [
                            [60.0, 122.0, 186.0, 192.0, 198.0, 134.0, 68.0],
                            [130.0, 264.0, 402.0, 414.0, 426.0, 288.0, 146.0],
                            [210.0, 426.0, 648.0, 666.0, 684.0, 462.0, 234.0],
                            [240.0, 486.0, 738.0, 756.0, 774.0, 522.0, 264.0],
                            [170.0, 344.0, 522.0, 534.0, 546.0, 368.0, 186.0],
                            [90.0, 182.0, 276.0, 282.0, 288.0, 194.0, 98.0],
                        ],
                        [
                            [40.0, 81.0, 123.0, 126.0, 129.0, 87.0, 44.0],
                            [85.0, 172.0, 261.0, 267.0, 273.0, 184.0, 93.0],
                            [135.0, 273.0, 414.0, 423.0, 432.0, 291.0, 147.0],
                            [150.0, 303.0, 459.0, 468.0, 477.0, 321.0, 162.0],
                            [105.0, 212.0, 321.0, 327.0, 333.0, 224.0, 113.0],
                            [55.0, 111.0, 168.0, 171.0, 174.0, 117.0, 59.0],
                        ],
                    ],
                ]
            ]
        ).astype(np.float32)

        return [y]

    op_tester.setPatterns(["ConvTranspose"], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, step_type="infer")


def test_convtranspose_pytorch(op_tester):
    def run_test(in_chans, out_chans, data, kernel):
        print(f"run_test({in_chans}, {out_chans}, {data}, {kernel})")
        assert len(data) == len(kernel)
        x = np.random.rand(1, in_chans, *data).astype(np.float32)
        W = np.random.rand(in_chans, out_chans, *kernel).astype(np.float32)

        def init_builder(builder):
            d = builder.addInputTensor(x)
            f = builder.addInputTensor(W)
            o = builder.aiOnnxOpset11.convtranspose([d, f])
            builder.addOutputTensor(o)
            return [o]

        def reference(_):  # ref_data is an unused argument
            data = torch.tensor(x)
            weights = torch.tensor(W)
            if len(kernel) == 1:
                conv = torch.nn.ConvTranspose1d(in_chans, out_chans, kernel)
            elif len(kernel) == 2:
                conv = torch.nn.ConvTranspose2d(in_chans, out_chans, kernel)
            else:
                raise SystemError(f"Bad kernel size {len(kernel)}")
            conv.weight.data = weights
            conv.bias.data = torch.zeros(conv.bias.size())
            o = conv(data)
            print(o.shape)
            return [o]

        op_tester.setPatterns(["ConvTranspose"], enableRuntimeAsserts=False)
        op_tester.run(init_builder, reference, step_type="infer")

    # Test various 2d convtransposes
    run_test(in_chans=1, out_chans=2, data=[3, 3], kernel=[3, 3])
    run_test(in_chans=1, out_chans=2, data=[4, 4], kernel=[4, 4])
    run_test(in_chans=1, out_chans=2, data=[5, 5], kernel=[5, 5])
    run_test(in_chans=1, out_chans=2, data=[4, 4], kernel=[3, 3])
    run_test(in_chans=1, out_chans=2, data=[5, 5], kernel=[3, 3])
    run_test(in_chans=1, out_chans=2, data=[5, 5], kernel=[4, 4])
    run_test(in_chans=2, out_chans=3, data=[3, 3], kernel=[3, 3])
    run_test(in_chans=3, out_chans=6, data=[4, 4], kernel=[4, 4])

    # Test various 1d convtransposes
    run_test(in_chans=1, out_chans=2, data=[3], kernel=[3])
    run_test(in_chans=1, out_chans=2, data=[4], kernel=[4])
    run_test(in_chans=1, out_chans=2, data=[5], kernel=[5])
    run_test(in_chans=1, out_chans=2, data=[4], kernel=[3])
    run_test(in_chans=1, out_chans=2, data=[5], kernel=[3])
    run_test(in_chans=1, out_chans=2, data=[5], kernel=[4])
    run_test(in_chans=2, out_chans=3, data=[3], kernel=[3])
    run_test(in_chans=3, out_chans=6, data=[4], kernel=[4])


def test_convtranspose_pytorch_attributes(op_tester):
    def run_test(
        in_chans,
        out_chans,
        data,
        kernel,
        groups=1,
        outshape=False,
        stride=None,
        output_padding=None,
        pads=None,
    ):
        print(f"run_test({in_chans}, {out_chans}, {data}, {kernel})")
        assert len(data) == len(kernel)
        x = np.random.rand(1, in_chans, *data).astype(np.float32)
        W = np.random.rand(in_chans, out_chans // groups, *kernel).astype(np.float32)
        bias = np.random.rand(out_chans).astype(np.float32)

        def reference(_):  # ref_data is an unused argument
            data = torch.tensor(x)
            weights = torch.tensor(W)

            kwargs = {}
            if stride:
                kwargs["stride"] = stride
            if output_padding:
                kwargs["output_padding"] = output_padding
            if pads:
                kwargs["padding"] = pads
            kwargs["groups"] = groups

            if len(kernel) == 1:
                conv = torch.nn.ConvTranspose1d(in_chans, out_chans, kernel, **kwargs)
            elif len(kernel) == 2:
                conv = torch.nn.ConvTranspose2d(in_chans, out_chans, kernel, **kwargs)
            else:
                raise SystemError(f"Bad kernel size {len(kernel)}")
            conv.weight.data = weights
            conv.bias.data = torch.tensor(bias)
            o = conv(data)
            print(o.shape)
            return [o]

        torch_out_shape = None
        if outshape:
            torch_out_shape = reference(x)[0].shape

        def init_builder(builder):
            d = builder.addInputTensor(x)
            f = builder.addInputTensor(W)
            b = builder.addInitializedInputTensor(bias)

            kwargs = {}
            if stride:
                kwargs["strides"] = stride
            if output_padding:
                kwargs["output_padding"] = output_padding
            if pads:
                kwargs["pads"] = pads + pads

            if torch_out_shape:
                kwargs["output_shape"] = torch_out_shape

            kwargs["group"] = groups
            o = builder.aiOnnxOpset10.convtranspose([d, f, b], **kwargs)
            builder.addOutputTensor(o)
            return [o]

        op_tester.setPatterns(popart.Patterns(popart.PatternsLevel.Default))
        op_tester.run(init_builder, reference, step_type="infer")

    # just testing strides
    run_test(in_chans=1, out_chans=2, data=[4], kernel=[4], stride=[5])
    run_test(in_chans=1, out_chans=2, data=[5], kernel=[4], stride=[5])
    run_test(in_chans=2, out_chans=3, data=[5], kernel=[4], stride=[5])
    run_test(in_chans=2, out_chans=3, data=[5, 5], kernel=[4, 4], stride=[3, 5])

    # testing output padding
    run_test(
        in_chans=1, out_chans=2, data=[4], kernel=[4], stride=[2], output_padding=[1]
    )

    # testing pads
    run_test(in_chans=1, out_chans=2, data=[3], kernel=[3], pads=[1])
    run_test(in_chans=1, out_chans=2, data=[5], kernel=[3], pads=[1])
    run_test(in_chans=1, out_chans=2, data=[5], kernel=[5], pads=[2])
    run_test(in_chans=1, out_chans=2, data=[5], kernel=[7], pads=[2])

    # deliberately excessive padding
    size_tuples = ((5, 3, 3), (9, 7, 7), (11, 9, 9))
    for d, k, p in size_tuples:
        for out_shape in (True, False):
            run_test(
                in_chans=1,
                out_chans=2,
                data=[d],
                kernel=[k],
                pads=[p],
                outshape=out_shape,
            )

    # stride and pads
    run_test(
        in_chans=1,
        out_chans=2,
        data=[3, 3],
        kernel=[3, 3],
        stride=[3, 2],
        pads=[1, 2],
        output_padding=[2, 1],
    )

    # Test groups
    run_test(in_chans=4, out_chans=4, data=[3], kernel=[3], pads=[1], groups=4)
    run_test(in_chans=8, out_chans=8, data=[5], kernel=[3], pads=[1], groups=2)

    # Test output shape
    run_test(in_chans=8, out_chans=8, data=[5], kernel=[3], pads=[1], outshape=True)


def test_where_0(op_tester):
    condition = np.array([[1, 0], [1, 1]], dtype=np.bool)
    x = np.array([[1, 2], [3, 4]], dtype=np.float32)
    y = np.array([[9, 8], [7, 6]], dtype=np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(condition)
        i2 = builder.addInputTensor(x)
        i3 = builder.addInputTensor(y)
        o = builder.aiOnnx.where([i1, i2, i3])
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        ct = torch.tensor(condition)
        cx = torch.tensor(x)
        cy = torch.tensor(y)
        out = torch.where(ct, cx, cy)
        return [out]

    op_tester.run(init_builder, reference, "infer")


@pytest.mark.parametrize("side", ("lhs", "rhs"))
def test_where_inplace(op_tester, side):
    lhs = side == "lhs"
    rhs = side == "rhs"
    assert lhs ^ rhs

    condition = np.random.rand(100, 200).astype(np.bool)
    x = np.random.rand(100, 200).astype(np.float32)
    y = np.random.rand(100, 200).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(condition)
        i2 = builder.addInputTensor(x)
        i3 = builder.addInputTensor(y)

        o = builder.aiOnnx.where([i1, i2, i3])

        builder.setInplacePreferences(
            o, {"WhereLhsInplace": 1 if lhs else 0, "WhereRhsInplace": 1 if rhs else 0}
        )

        builder.addOutputTensor(i2)
        builder.addOutputTensor(i3)
        builder.addOutputTensor(o)

        return [o]

    def reference(_):  # ref_data is an unused argument
        ct = torch.tensor(condition)
        cx = torch.tensor(x).float()
        cy = torch.tensor(y).float()
        out = torch.where(ct, cx, cy)
        return [out]

    patterns = ["InPlace"]
    op_tester.setPatterns(patterns, enableRuntimeAsserts=False)

    _ = op_tester.run(init_builder, reference, "infer")


def test_round(op_tester):
    d1 = np.random.rand(2, 7).astype(np.float32)
    d1 = d1 * 6 - 3  # numbers in range [-3, 3]

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnxOpset11.round([i1], "test_round")
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        result = np.round(d1)
        return [result.astype(np.float32)]

    op_tester.run(init_builder, reference, "infer")


def test_round_graphcore(op_tester):
    d1 = np.random.rand(2, 7).astype(np.float32)
    d1 = d1 * 6 - 3  # numbers in range [-3, 3]

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiGraphcore.round([i1], "test_round")
        builder.addOutputTensor(o)
        # Check builder shape inference
        assert builder.getTensorShape(o) == [2, 7]
        return [o]

    def reference(_):  # ref_data is an unused argument
        result = np.round(d1)
        return [result.astype(np.float32)]

    op_tester.inplacing = False
    op_tester.run(init_builder, reference, "infer")


def test_round_inplace(op_tester):
    d1 = np.random.rand(2, 7).astype(np.float32)
    d1 = d1 * 10  # numbers in range [0, 10]

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        # Pad with ops to allow in-placing
        log = builder.aiOnnxOpset11.log([i1])
        round = builder.aiOnnxOpset11.round([log], "test_round")
        o = builder.aiOnnxOpset11.exp([round])
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        result = np.exp(np.round(np.log(d1)))
        return [result.astype(np.float32)]

    op_tester.setPatterns(["InPlace"], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, "train")


def test_round_grad(op_tester):
    d1 = np.random.rand(2, 7).astype(np.float32)
    d1 = d1 * 6 - 3  # numbers in range [-3, 3]

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnxOpset11.round([i1], "test_round")
        builder.addOutputTensor(o)
        return [o, popart.reservedGradientPrefix() + i1]

    def reference(_):  # ref_data is an unused argument
        return [np.round(d1).astype(np.float32), np.zeros_like(d1)]

    op_tester.run(init_builder, reference, "train")


def test_where_1(op_tester):
    condition = np.array([[1, 0], [1, 1]], dtype=np.bool)
    x = np.array([[1, 2]], dtype=np.float32)
    y = np.array([[9, 8], [7, 6]], dtype=np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(condition)
        i2 = builder.addInputTensor(x)
        i3 = builder.addInputTensor(y)
        o = builder.aiOnnx.where([i1, i2, i3])
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        ct = torch.tensor(condition)
        cx = torch.tensor(x)
        cy = torch.tensor(y)
        out = torch.where(ct, cx, cy)
        return [out]

    op_tester.run(init_builder, reference, "infer")


def test_where_2(op_tester):
    x = np.arange(9, dtype=np.int32)
    y = 10 * x
    condition = x < 5

    def init_builder(builder):
        i1 = builder.addInputTensor(condition)
        i2 = builder.addInputTensor(x)
        i3 = builder.addInputTensor(y)
        o = builder.aiOnnx.where([i1, i2, i3])
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        ct = torch.tensor(condition)
        cx = torch.tensor(x)
        cy = torch.tensor(y)
        out = torch.where(ct, cx, cy)
        return [out]

    op_tester.run(init_builder, reference, "infer")


def test_where_3(op_tester):
    x, y = np.ogrid[:3, :4]
    x = np.float32(x)
    y = np.float32(y)
    y = 10 + y
    condition = x < y

    def init_builder(builder):
        i1 = builder.addInputTensor(condition)
        i2 = builder.addInputTensor(x)
        i3 = builder.addInputTensor(y)
        o = builder.aiOnnx.where([i1, i2, i3])
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        ct = torch.tensor(condition)
        cx = torch.tensor(x)
        cy = torch.tensor(y)
        out = torch.where(ct, cx, cy)
        return [out]

    op_tester.run(init_builder, reference, "infer")


def test_where_4(op_tester):
    x = np.array([[0, 1, 2], [0, 2, 4], [0, 3, 6]])
    y = np.array([-1])
    x = np.float32(x)
    y = np.float32(y)
    condition = x < 4

    def init_builder(builder):
        i1 = builder.addInputTensor(condition)
        i2 = builder.addInputTensor(x)
        i3 = builder.addInputTensor(y)
        o = builder.aiOnnx.where([i1, i2, i3])
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        ct = torch.tensor(condition)
        cx = torch.tensor(x)
        cy = torch.tensor(y)
        out = torch.where(ct, cx, cy)
        return [out]

    op_tester.run(init_builder, reference, "infer")


def test_where_5(op_tester):
    x = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 10])
    x = np.float32(x)
    y = np.float32(y)
    condition = np.array([False])

    def init_builder(builder):
        i1 = builder.addInputTensor(condition)
        i2 = builder.addInputTensor(x)
        i3 = builder.addInputTensor(y)
        o = builder.aiOnnx.where([i1, i2, i3])
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        ct = torch.tensor(condition)
        cx = torch.tensor(x)
        cy = torch.tensor(y)
        out = torch.where(ct, cx, cy)
        return [out]

    op_tester.run(init_builder, reference, "infer")


def test_where_grad0(op_tester):
    condition = np.array([[True, False], [True, True]], dtype=np.bool)
    x = np.array([[1, 2], [3, 4]], dtype=np.float32)
    y = np.array([[9, 8], [7, 6]], dtype=np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(condition)
        i2 = builder.addInputTensor(x)
        i3 = builder.addInputTensor(y)
        o = builder.aiOnnx.where([i1, i2, i3])
        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + i2,
            popart.reservedGradientPrefix() + i3,
            popart.reservedGradientPrefix() + o,
        ]

    def reference(ref_data):
        tc = torch.tensor(condition)
        tx = torch.tensor(x, requires_grad=True)
        ty = torch.tensor(y, requires_grad=True)
        out = torch.where(tc, tx, ty)
        d__o = ref_data.getOutputTensorGrad(0)
        out.backward(torch.tensor(d__o))
        return [out, tx.grad, ty.grad, None]

    op_tester.run(init_builder, reference, "train")


def test_where_grad1(op_tester):
    condition = np.array([[True, False], [True, True]], dtype=np.bool)
    x = np.array([[1, 2]], dtype=np.float32)
    y = np.array([[9, 8], [7, 6]], dtype=np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(condition)
        i2 = builder.addInputTensor(x)
        i3 = builder.addInputTensor(y)
        o = builder.aiOnnx.where([i1, i2, i3])
        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + i2,
            popart.reservedGradientPrefix() + i3,
            popart.reservedGradientPrefix() + o,
        ]

    def reference(ref_data):
        tc = torch.tensor(condition)
        tx = torch.tensor(x, requires_grad=True)
        ty = torch.tensor(y, requires_grad=True)
        out = torch.where(tc, tx, ty)
        d__o = ref_data.getOutputTensorGrad(0)
        out.backward(torch.tensor(d__o))
        return [out, tx.grad, ty.grad, None]

    op_tester.run(init_builder, reference, "train")


def test_where_grad2(op_tester):
    x = np.arange(9, dtype=np.float32)
    y = 10 * x
    condition = x < 5

    def init_builder(builder):
        i1 = builder.addInputTensor(condition)
        i2 = builder.addInputTensor(x)
        i3 = builder.addInputTensor(y)
        o = builder.aiOnnx.where([i1, i2, i3])
        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + i2,
            popart.reservedGradientPrefix() + i3,
            popart.reservedGradientPrefix() + o,
        ]

    def reference(ref_data):
        tc = torch.tensor(condition)
        tx = torch.tensor(x, requires_grad=True)
        ty = torch.tensor(y, requires_grad=True)
        out = torch.where(tc, tx, ty)
        d__o = ref_data.getOutputTensorGrad(0)
        out.backward(torch.tensor(d__o))
        return [out, tx.grad, ty.grad, None]

    op_tester.run(init_builder, reference, "train")


def test_where_grad3(op_tester):
    x, y = np.ogrid[:3, :4]
    x = np.float32(x)
    y = np.float32(y)
    y = 10 + y
    condition = x < y

    def init_builder(builder):
        i1 = builder.addInputTensor(condition)
        i2 = builder.addInputTensor(x)
        i3 = builder.addInputTensor(y)
        o = builder.aiOnnx.where([i1, i2, i3])
        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + i2,
            popart.reservedGradientPrefix() + i3,
            popart.reservedGradientPrefix() + o,
        ]

    def reference(ref_data):
        tc = torch.tensor(condition)
        tx = torch.tensor(x, requires_grad=True)
        ty = torch.tensor(y, requires_grad=True)
        out = torch.where(tc, tx, ty)
        d__o = ref_data.getOutputTensorGrad(0)
        out.backward(torch.tensor(d__o))
        return [out, tx.grad, ty.grad, None]

    op_tester.run(init_builder, reference, "train")


def test_where_grad4(op_tester):
    x = np.array([[0, 1, 2], [0, 2, 4], [0, 3, 6]])
    y = np.array([-1])
    x = np.float32(x)
    y = np.float32(y)
    condition = x < 4

    def init_builder(builder):
        i1 = builder.addInputTensor(condition)
        i2 = builder.addInputTensor(x)
        i3 = builder.addInputTensor(y)
        o = builder.aiOnnx.where([i1, i2, i3])
        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + i2,
            popart.reservedGradientPrefix() + i3,
            popart.reservedGradientPrefix() + o,
        ]

    def reference(ref_data):
        tc = torch.tensor(condition)
        tx = torch.tensor(x, requires_grad=True)
        ty = torch.tensor(y, requires_grad=True)
        out = torch.where(tc, tx, ty)
        d__o = ref_data.getOutputTensorGrad(0)
        out.backward(torch.tensor(d__o))
        return [out, tx.grad, ty.grad, None]

    op_tester.run(init_builder, reference, "train")


def test_where_grad5(op_tester):
    x = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 10])
    x = np.float32(x)
    y = np.float32(y)
    condition = np.array([False])

    def init_builder(builder):
        i1 = builder.addInputTensor(condition)
        i2 = builder.addInputTensor(x)
        i3 = builder.addInputTensor(y)
        o = builder.aiOnnx.where([i1, i2, i3])
        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + i2,
            popart.reservedGradientPrefix() + i3,
            popart.reservedGradientPrefix() + o,
        ]

    def reference(ref_data):
        tc = torch.tensor(condition)
        tx = torch.tensor(x, requires_grad=True)
        ty = torch.tensor(y, requires_grad=True)
        out = torch.where(tc, tx, ty)
        d__o = ref_data.getOutputTensorGrad(0)
        out.backward(torch.tensor(d__o))
        return [out, tx.grad, ty.grad, None]

    op_tester.run(init_builder, reference, "train")


@pytest.mark.parametrize("npType", [np.int32, np.uint32])
def test_bitwise_not(op_tester, npType):
    d1 = np.random.uniform(-10, 10, 10).astype(npType)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiGraphcore.bitwisenot([i1], "test_bitwisenot")
        builder.addOutputTensor(o)
        # Check builder shape inference
        assert builder.getTensorShape(o) == [10]
        return [o]

    def reference(_):  # ref_data is an unused argument
        out = np.bitwise_not(d1)
        return [out]

    op_tester.run(init_builder, reference, "infer")


@pytest.mark.parametrize("npType", [np.int32, np.uint32])
@pytest.mark.parametrize("npOp", [np.bitwise_and, np.bitwise_or, np.bitwise_xor])
def test_bitwise_binary_op(op_tester, npType, npOp):
    d1 = np.random.uniform(-10, 10, 10).astype(npType)
    d2 = np.random.uniform(-10, 10, 10).astype(npType)

    def init_builder(builder):
        opname = npOp.__name__.replace("_", "")
        gcOp = getattr(builder.aiGraphcore, opname)
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        o = gcOp([i1, i2], "test_bitwise_binary_op")
        builder.addOutputTensor(o)
        # Check builder shape inference
        assert builder.getTensorShape(o) == [10]
        return [o]

    def reference(_):  # ref_data is an unused argument
        out = npOp(d1, d2)
        return [out]

    op_tester.run(init_builder, reference, "infer")
