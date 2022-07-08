# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import numpy as np
import pytest
import popart
import torch
import torch.nn.functional as F
import test_util as tu
from packaging import version
import numbers

import onnx.backend.test.case.node.resize as onnx_resize


# For debugging failing tests, it can be useful to use known data, rather than random.
# This is just a utility for easily swapping a call like
#     data = np.random.rand(1, 1, *data_shape).astype(np.float32)
# with
#     data = arange([1, 1] + data_shape)
def arange(shape):
    x = np.arange(np.prod(shape)).astype(np.float32)
    return np.reshape(x, shape)


# if the version of torch is greater or equal to 1.5.0, use
# F.interpolate, otherwise use a matching interpolate
# function. This is required as torch versions below 1.5.0
# don't have the `recompute_scale_factor` parameter for
# `F.interpolate` and not all the buildbots appear to have
# an up to date version of torch.
def interpolate(data, scale_factor):
    if version.parse(torch.__version__) >= version.parse("1.5.0"):
        return F.interpolate(
            data, scale_factor=scale_factor, recompute_scale_factor=False
        )
    else:
        if isinstance(scale_factor, numbers.Number):
            scale_factor = [scale_factor]
        scale_factor = [1.0, 1.0] + scale_factor

        result = data
        out_shape = data.shape
        out_shape = [int(i * s) for i, s in zip(out_shape, scale_factor)]

        def resize_nearest(x, dim, size, scale):
            slices = torch.split(x, 1, dim)
            to_concat = []

            to_concat = [slices[int(i / scale)] for i in range(size)]

            return torch.cat(to_concat, dim)

        for i in range(len(out_shape)):
            if data.shape[i] != out_shape[i]:
                result = resize_nearest(result, i, out_shape[i], scale_factor[i])

        return result


@pytest.mark.parametrize(
    "data_shape, scales",
    [
        # upsample
        ([2, 2], [2.0, 3.0]),
        # downsample
        ([2, 4], [0.5, 0.5]),
    ],
)
def test_resize_10(op_tester, data_shape, scales):

    data = np.random.rand(1, 1, *data_shape).astype(np.float32)

    scales = np.array([1.0, 1.0] + scales, dtype=np.float32)

    def init_builder(builder):
        d = builder.addInputTensor(data)
        s = builder.aiOnnx.constant(scales)
        o = builder.aiOnnx.resize([d, s])
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        o = onnx_resize.interpolate_nd(
            data, onnx_resize.nearest_coeffs, scale_factors=scales
        )
        return [o.astype(data.dtype)]

    op_tester.run(init_builder, reference, "infer")


@pytest.mark.parametrize(
    "scale_factor",
    [
        (2),
        (3),
        (0.5),
        (0.25),
    ],
)
def test_resize_nearest_grad_1d(op_tester, scale_factor):
    data = np.array([[[1, 2, 3, 4]]], dtype=np.float32)

    # This is to weight the gradient
    x_data = [2 ** i for i in range(int(4 * scale_factor))]
    x_data = np.array([[x_data]], dtype=np.float32)

    scales = np.array([1.0, 1.0, float(scale_factor)], dtype=np.float32)

    def init_builder(builder):
        x = builder.addInputTensor(x_data)
        d = builder.addInputTensor(data)
        s = builder.aiOnnx.constant(scales)
        o = builder.aiOnnx.resize([d, s])
        o = builder.aiOnnx.mul([o, x])
        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + d,
            popart.reservedGradientPrefix() + o,
        ]

    def reference(ref_data):
        a = torch.tensor(data, requires_grad=True)
        b = interpolate(a, scale_factor)
        b.retain_grad()
        o = b * torch.tensor(x_data)

        d__o = ref_data.getOutputTensorGrad(0)
        o.backward(torch.tensor(d__o))
        print("-" * 60)
        print(d__o)
        print(b.grad)
        print(a.grad)
        print("-" * 60)
        return [o, a.grad, None]

    op_tester.setPatterns(["MulArgGradOp"], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, "train")


@pytest.mark.parametrize(
    "factor1, factor2",
    [
        (2, 3),
        (0.5, 0.25),
        (2, 0.5),
    ],
)
def test_resize_nearest_grad_2d(op_tester, factor1, factor2):

    data = np.array(
        [
            [
                [
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                ]
            ]
        ],
        dtype=np.float32,
    )

    x_data = [2 ** i for i in range(int(2 * factor1) * int(4 * factor2))]
    x_data = np.reshape(
        np.array([x_data], dtype=np.float32), [1, 1, int(2 * factor1), int(4 * factor2)]
    )

    scales = np.array([1.0, 1.0, float(factor1), float(factor2)], dtype=np.float32)

    def init_builder(builder):
        d = builder.addInputTensor(data)
        x = builder.addInputTensor(x_data)
        s = builder.aiOnnx.constant(scales)
        o = builder.aiOnnx.resize([d, s])
        o = builder.aiOnnx.mul([o, x])
        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + d,
            popart.reservedGradientPrefix() + o,
        ]

    def reference(ref_data):
        a = torch.tensor(data, requires_grad=True)
        b = interpolate(a, [factor1, factor2])
        b.retain_grad()
        o = b * torch.tensor(x_data)

        d__o = ref_data.getOutputTensorGrad(0)
        o.backward(torch.tensor(d__o))
        print("-" * 60)
        print(d__o)
        print(b.grad)
        print(a.grad)
        print("-" * 60)
        return [o, a.grad, None]

    op_tester.setPatterns(["MulArgGradOp"], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, "train")


@tu.requires_ipu_model
@pytest.mark.parametrize(
    "data_shape, scales",
    [
        # upsample
        ([5, 3], [2.3, 2.5]),
        # downsample
        ([2, 8], [1.0, 3 / 8]),
        ([5, 3], [0.3, 0.5]),
    ],
)
def test_nearest_grad(op_tester, data_shape, scales):
    data = np.random.rand(1, 1, *data_shape).astype(np.float32)

    scales = np.array([1.0, 1.0] + scales, dtype=np.float32)

    x_data_shape = [int(i * j) for i, j in zip(data.shape, scales)]
    x_data = np.random.rand(*x_data_shape).astype(np.float32)

    def init_builder(builder):
        d = builder.addInputTensor(data)
        x = builder.addInputTensor(x_data)
        s = builder.aiOnnx.constant(scales)
        o = builder.aiOnnx.resize([d, s])
        o = builder.aiOnnx.mul([o, x])
        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + d,
            popart.reservedGradientPrefix() + o,
        ]

    def reference(ref_data):
        a = torch.tensor(data, requires_grad=True)
        s = [i for i in scales[2:]]
        b = interpolate(a, s)
        b.retain_grad()
        o = b * torch.tensor(x_data)

        d__o = ref_data.getOutputTensorGrad(0)
        o.backward(torch.tensor(d__o))
        return [o, a.grad, None]

    op_tester.setPatterns(["MulArgGradOp"], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, "train")


@pytest.mark.parametrize(
    "data_shape, scales",
    [
        # This changes the values without changing the size of the dimension.
        ([8], [1.12]),
        # upsample
        ([2], [8.0]),
        ([5, 3], [2.3, 2.5]),
        # downsample
        ([2, 8], [1.0, 3 / 8]),
        ([5, 3], [0.3, 0.5]),
    ],
)
@pytest.mark.parametrize(
    "nearest_mode",
    ["round_prefer_floor", "round_prefer_ceil", "floor", "ceil", "pytorch"],
)
@pytest.mark.parametrize(
    "coordinate_transformation_mode",
    ["half_pixel", "pytorch_half_pixel", "asymmetric", "align_corners"],
)
def test_resize_11(
    op_tester, data_shape, scales, nearest_mode, coordinate_transformation_mode
):
    if nearest_mode == "pytorch":
        data_shape = [1, 1] + data_shape
        scales = [1.0, 1.0] + scales

        # Known PyTorch issue
        # https://github.com/pytorch/pytorch/issues/62237
        if torch.__version__.startswith("1.9"):
            if scales[-1] == 1.12:
                pytest.skip()

    data = np.random.rand(*data_shape).astype(np.float32)
    roi = np.array([], dtype=np.float32)
    scales = np.array(scales, dtype=np.float32)

    def init_builder(builder):
        d = builder.addInputTensor(data)
        s = builder.aiOnnxOpset11.constant(scales, False)
        r = builder.aiOnnxOpset11.constant(roi, False)
        o = builder.aiOnnxOpset11.resize(
            [d, r, s],
            nearest_mode=nearest_mode,
            coordinate_transformation_mode=coordinate_transformation_mode,
        )
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        if nearest_mode == "pytorch":
            x = torch.tensor(data)
            s = [i for i in scales[2:]]
            o = interpolate(x, s)
            return [o]
        else:

            def coeffs(ratio):
                return onnx_resize.nearest_coeffs(ratio, mode=nearest_mode)

            o = onnx_resize.interpolate_nd(
                data,
                coeffs,
                scale_factors=scales,
                coordinate_transformation_mode=coordinate_transformation_mode,
            )
            return [o.astype(data.dtype)]

    op_tester.run(init_builder, reference, "infer")


def test_resize_11_debug():
    data = np.random.rand(1, 1, 2, 2).astype(np.float32)
    roi = np.array([], dtype=np.float32)
    scales = np.array([1.0, 1.0, 2.0, 3.0], dtype=np.float32)

    builder = popart.Builder()
    d = builder.addInputTensor(popart.TensorInfo("FLOAT", [1, 1, 2, 2]))
    s = builder.aiOnnxOpset11.constant(scales, False)
    r = builder.aiOnnxOpset11.constant(roi, False)
    o = builder.aiOnnxOpset11.resize([d, r, s])
    builder.addOutputTensor(o)

    proto = builder.getModelProto()
    print(f"Proto: {proto}")

    dataFlow = popart.DataFlow(1, {o: popart.AnchorReturnType("All")})

    with tu.create_test_device() as device:
        print("Creating session")
        sess = popart.InferenceSession(proto, dataFlow, device)

        print("Initializing anchor arrays")
        anchors = sess.initAnchorArrays()

        print("Preparinng device")
        sess.prepareDevice()

        print("Creating stepio")
        inputs = {d: data}
        stepio = popart.PyStepIO(inputs, anchors)

        print("Running model")
        sess.run(stepio)

        print("Fin")
