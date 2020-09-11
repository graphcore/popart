# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import numpy as np
import pytest
import popart
import torch
import torch.nn.functional as F
from op_tester import op_tester
import test_util as tu
import os
os.environ['POPART_LOG_LEVEL'] = 'TRACE'


def test_upsample_nearest(op_tester):
    def run_test(data_shape, scales):
        data = np.random.rand(1, 1, *data_shape).astype(np.float32)

        scales = np.array([1.0, 1.0] + scales, dtype=np.float32)

        def init_builder(builder):
            d = builder.addInputTensor(data)
            s = builder.aiOnnx.constant(scales)
            o = builder.aiOnnx.resize([d, s])
            builder.addOutputTensor(o)
            return [o]

        def reference(ref_data):
            x = torch.tensor(data)
            s = [
                int(i * scale) for i, scale in zip(data.shape[2:], scales[2:])
            ]
            o = F.interpolate(x, s)
            return [o]

        op_tester.run(init_builder, reference, 'infer')

    run_test([2, 2], [2.0, 3.0])
    run_test([2, 2], [2.5, 2.5])
    run_test([3, 2], [2.5, 2.5])
    run_test([5, 3], [2.3, 2.5])


def test_downsample_nearest(op_tester):
    def run_test(data_shape, scales):
        data = np.random.rand(1, 1, *data_shape).astype(np.float32)

        scales = np.array([1.0, 1.0] + scales, dtype=np.float32)

        def init_builder(builder):
            d = builder.addInputTensor(data)
            s = builder.aiOnnx.constant(scales)
            o = builder.aiOnnx.resize([d, s])
            builder.addOutputTensor(o)
            return [o]

        def reference(ref_data):
            x = torch.tensor(data)
            s = [
                int(i * scale) for i, scale in zip(data.shape[2:], scales[2:])
            ]
            o = F.interpolate(x, s)
            return [o]

        op_tester.run(init_builder, reference, 'infer')

    run_test([2, 4], [0.5, 0.5])
    run_test([2, 8], [1.0, 3 / 8])
    run_test([5, 4], [0.5, 0.5])
    run_test([5, 3], [0.3, 0.5])


def test_resize_nearest_grad_1d(op_tester):
    def run_test(scale_factor):
        data = np.array([[[1, 2, 3, 4]]], dtype=np.float32)

        # This is to weight the gradient
        x_data = [2**i for i in range(int(4 * scale_factor))]
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
            b = F.interpolate(a, [int(4 * scale_factor)])
            b.retain_grad()
            o = b * torch.tensor(x_data)

            d__o = ref_data.getOutputTensorGrad(0)
            o.backward(torch.tensor(d__o))
            print('-' * 60)
            print(d__o)
            print(b.grad)
            print(a.grad)
            print('-' * 60)
            return [o, a.grad, None]

        op_tester.setPatterns(['MulArgGradOp'], enableRuntimeAsserts=False)
        op_tester.run(init_builder, reference, 'train')

    run_test(2)
    run_test(3)
    run_test(0.5)
    run_test(0.25)


def test_resize_nearest_grad_2d(op_tester):
    def run_test(factor1, factor2):
        data = np.array([[[
            [1, 2, 3, 4],
            [5, 6, 7, 8],
        ]]], dtype=np.float32)

        x_data = [2**i for i in range(int(2 * factor1) * int(4 * factor2))]
        x_data = np.reshape(
            np.array([x_data], dtype=np.float32),
            [1, 1, int(2 * factor1), int(4 * factor2)])

        scales = np.array([1.0, 1.0, float(factor1),
                           float(factor2)],
                          dtype=np.float32)

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
            b = F.interpolate(a, [int(2 * factor1), int(4 * factor2)])
            b.retain_grad()
            o = b * torch.tensor(x_data)

            d__o = ref_data.getOutputTensorGrad(0)
            o.backward(torch.tensor(d__o))
            print('-' * 60)
            print(d__o)
            print(b.grad)
            print(a.grad)
            print('-' * 60)
            return [o, a.grad, None]

        op_tester.setPatterns(['MulArgGradOp'], enableRuntimeAsserts=False)
        op_tester.run(init_builder, reference, 'train')

    run_test(2, 2)
    run_test(2, 3)
    run_test(3, 2)

    run_test(0.5, 0.5)
    run_test(0.5, 0.25)

    run_test(2, 0.5)
    run_test(0.5, 2)


def test_upsample_nearest_grad(op_tester):
    def run_test(data_shape, scales):
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
            s = [
                int(i * scale) for i, scale in zip(data.shape[2:], scales[2:])
            ]
            b = F.interpolate(a, s)
            b.retain_grad()
            o = b * torch.tensor(x_data)

            d__o = ref_data.getOutputTensorGrad(0)
            o.backward(torch.tensor(d__o))
            return [o, a.grad, None]

        op_tester.setPatterns(['MulArgGradOp'], enableRuntimeAsserts=False)
        op_tester.run(init_builder, reference, 'train')

    run_test([2, 2], [2.0, 3.0])
    run_test([2, 2], [2.5, 2.5])
    run_test([3, 2], [2.5, 2.5])
    run_test([5, 3], [2.3, 2.5])


def test_downsample_nearest_grad(op_tester):
    def run_test(data_shape, scales):
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
            s = [
                int(i * scale) for i, scale in zip(data.shape[2:], scales[2:])
            ]
            b = F.interpolate(a, s)
            b.retain_grad()
            o = b * torch.tensor(x_data)

            d__o = ref_data.getOutputTensorGrad(0)
            o.backward(torch.tensor(d__o))
            return [o, a.grad, None]

        op_tester.setPatterns(['MulArgGradOp'], enableRuntimeAsserts=False)
        op_tester.run(init_builder, reference, 'train')

    run_test([2, 4], [0.5, 0.5])
    run_test([2, 8], [1.0, 3 / 8])
    run_test([5, 4], [0.5, 0.5])
    run_test([5, 3], [0.3, 0.5])


def test_resize_11(op_tester):
    data = np.random.rand(1, 1, 2, 2).astype(np.float32)
    roi = np.array([], dtype=np.float32)
    scales = np.array([1.0, 1.0, 2.0, 3.0], dtype=np.float32)

    def init_builder(builder):
        d = builder.addInputTensor(data)
        s = builder.aiOnnxOpset11.constant(scales, False)
        r = builder.aiOnnxOpset11.constant(roi, False)
        o = builder.aiOnnxOpset11.resize([d, r, s])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        x = torch.tensor(data)
        s = [int(i * scale) for i, scale in zip(data.shape[2:], scales[2:])]
        o = F.interpolate(x, s)
        return [o]

    op_tester.run(init_builder, reference, 'infer')


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
    print(f'Proto: {proto}')

    dataFlow = popart.DataFlow(1, {o: popart.AnchorReturnType("All")})

    print('Creating session')
    sess = popart.InferenceSession(proto, dataFlow, tu.create_test_device())

    print(f'Initializing anchor arrays')
    anchors = sess.initAnchorArrays()

    print(f'Preparinng device')
    sess.prepareDevice()

    print(f'Creating stepio')
    inputs = {d: data}
    stepio = popart.PyStepIO(inputs, anchors)

    print(f'Running model')
    sess.run(stepio)

    print(f'Fin')


def test_float16_scales(op_tester):
    data = np.random.rand(1, 1, 2, 2).astype(np.float32)

    scales = np.array([1.0, 1.0, 2.0, 3.0], dtype=np.float16)

    def init_builder(builder):
        d = builder.addInputTensor(data)
        s = builder.aiOnnx.constant(scales)
        o = builder.aiOnnx.resize([d, s])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        x = torch.tensor(data)
        s = [int(i * scale) for i, scale in zip(data.shape[2:], scales[2:])]
        o = F.interpolate(x, s)
        return [o]

    op_tester.run(init_builder, reference, 'infer')


def test_resize11_float16_scales(op_tester):
    data = np.random.rand(1, 1, 2, 2).astype(np.float32)

    roi = np.array([], dtype=np.float32)
    scales = np.array([1.0, 1.0, 2.0, 3.0], dtype=np.float16)

    def init_builder(builder):
        d = builder.addInputTensor(data)
        r = builder.aiOnnxOpset11.constant(roi, False)
        s = builder.aiOnnxOpset11.constant(scales)
        o = builder.aiOnnxOpset11.resize([d, r, s])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        x = torch.tensor(data)
        s = [int(i * scale) for i, scale in zip(data.shape[2:], scales[2:])]
        o = F.interpolate(x, s)
        return [o]

    op_tester.run(init_builder, reference, 'infer')
