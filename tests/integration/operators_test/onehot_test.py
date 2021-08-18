# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import numpy as np
import pytest
import popart
import torch
from op_tester import op_tester


def test_onehot_2d_with_axis_testing(op_tester):
    indices = np.array([1, 5]).astype(np.int32)
    depth = np.array(6).astype(np.int32)
    values = np.array([0, 1]).astype(np.int32)

    output = np.array([[0, 0], [1, 0], [0, 0], [0, 0], [0, 0],
                       [0, 1]]).astype(np.int32)

    def init_builder(builder):
        print(depth)
        i1 = builder.addInputTensor(indices)
        i2 = builder.aiOnnx.constant(depth)  # depth has to be a constant
        i3 = builder.addInputTensor(values)
        o = builder.aiOnnx.onehot([i1, i2, i3], 0, "test_onehot")
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        return [output]

    op_tester.setPatterns(['OpToIdentity'], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'infer')


def test_onehot_2d_without_axis_testing(op_tester):
    indices = np.array([1, 5]).astype(np.int32)
    depth = np.array(6).astype(np.int64)
    values = np.random.rand(2).astype(np.float32)

    output = np.array(
        [[values[0], values[1], values[0], values[0], values[0], values[0]],
         [values[0], values[0], values[0], values[0], values[0],
          values[1]]]).astype(np.float32)

    def init_builder(builder):
        print(depth)
        i1 = builder.addInputTensor(indices)
        i2 = builder.aiOnnx.constant(depth)  # depth has to be a constant
        i3 = builder.addInputTensor(values)
        o = builder.aiOnnx.onehot([i1, i2, i3], -1, "test_onehot")
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        return [output]

    op_tester.setPatterns(['OpToIdentity'], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'infer')


def test_onehot_2d_with_axis_training(op_tester):
    indices = np.array([1, 5]).astype(np.int32)
    depth = np.array(6).astype(np.int16)
    values = np.array([-0.5, 0.5]).astype(np.float32)

    output = np.array([[values[0], values[0]], [values[1], values[0]],
                       [values[0], values[0]], [values[0], values[0]],
                       [values[0], values[0]], [values[0],
                                                values[1]]]).astype(np.float32)

    output_grad = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0],
                            [1.0, 1.0], [1.0, 1.0]]).astype(np.float32)

    values_grad = np.array([10.0, 2.0]).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(indices)
        i2 = builder.aiOnnx.constant(depth)  # depth has to be a constant
        i3 = builder.addInputTensor(values)
        o = builder.aiOnnx.onehot([i1, i2, i3], 0, "test_onehot")
        builder.addOutputTensor(o)
        return [
            o,
            popart.TensorId(popart.reservedGradientPrefix() + o),
            popart.TensorId(popart.reservedGradientPrefix() + i3)
        ]

    def reference(ref_data):
        return [output, output_grad, values_grad]

    op_tester.lossReduction = popart.ReductionType.Sum
    op_tester.setPatterns(['OpToIdentity'], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'train')


def test_onehot_2d_without_axis_training(op_tester):
    indices = np.array([1, 5]).astype(np.int32)
    depth = np.array(6).astype(np.uint8)
    values = np.array([-0.5, 0.5]).astype(np.float32)

    output = np.array(
        [[values[0], values[1], values[0], values[0], values[0], values[0]],
         [values[0], values[0], values[0], values[0], values[0],
          values[1]]]).astype(np.float32)

    output_grad = np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]).astype(np.float32)

    values_grad = np.array([10.0, 2.0]).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(indices)
        i2 = builder.aiOnnx.constant(depth)  # depth has to be a constant
        i3 = builder.addInputTensor(values)
        o = builder.aiOnnx.onehot([i1, i2, i3], -1, "test_onehot")
        builder.addOutputTensor(o)
        return [
            o,
            popart.TensorId(popart.reservedGradientPrefix() + o),
            popart.TensorId(popart.reservedGradientPrefix() + i3)
        ]

    def reference(ref_data):
        return [output, output_grad, values_grad]

    op_tester.lossReduction = popart.ReductionType.Sum
    op_tester.setPatterns(['OpToIdentity'], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'train')


def test_onehot_3d_without_axis_testing(op_tester):
    indices = np.array([[1, 2], [2, 1]]).astype(np.int32)
    depth = np.array(4).astype(np.float32)
    values = np.array([0, 1]).astype(np.int32)

    output = np.array([[[0, 1, 0, 0], [0, 0, 1, 0]],
                       [[0, 0, 1, 0], [0, 1, 0, 0]]]).astype(np.int32)

    def init_builder(builder):
        print(depth)
        i1 = builder.addInputTensor(indices)
        i2 = builder.aiOnnx.constant(depth)  # depth has to be a constant
        i3 = builder.addInputTensor(values)
        o = builder.aiOnnx.onehot([i1, i2, i3], -1, "test_onehot")
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        return [output]

    op_tester.setPatterns(['OpToIdentity'], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'infer')


def test_onehot_3d_with_axis_testing(op_tester):
    indices = np.array([[1, 2], [2, 1]]).astype(np.int32)
    depth = np.array(4).astype(np.int8)
    values = np.array([0, 1]).astype(np.int32)

    output = np.array([[[0, 0], [1, 0], [0, 1], [0, 0]],
                       [[0, 0], [0, 1], [1, 0], [0, 0]]]).astype(np.int32)

    def init_builder(builder):
        print(depth)
        i1 = builder.addInputTensor(indices)
        i2 = builder.aiOnnx.constant(depth)  # depth has to be a constant
        i3 = builder.addInputTensor(values)
        o = builder.aiOnnx.onehot([i1, i2, i3], 1, "test_onehot")
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        return [output]

    op_tester.setPatterns(['OpToIdentity'], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'infer')
