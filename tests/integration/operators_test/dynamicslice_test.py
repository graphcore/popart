# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import itertools
import numpy as np
import popart
import torch
import pytest
import torch.nn.functional as F
from op_tester import op_tester


# Test a set of non-overlapping dynamic slices
# tensor -> S0 -> out0
# tensor -> S1 -> out1
# tensor -> S2 -> out2
# where out0, out1 and out2 are non-overlapping subregions of the tensor
def test_dynamicslice(op_tester):
    data = np.random.rand(5, 12, 7).astype(np.float32)
    axes = [1]
    sizes = [3]

    def init_builder(builder):
        tensor = builder.addInputTensor(data)
        result = []
        for sliceid in range(4):
            index = builder.addInputTensor(np.asarray([sliceid * 3],
                                                      np.uint32))
            out = builder.aiGraphcore.dynamicslice([tensor, index],
                                                   axes=axes,
                                                   sizes=sizes,
                                                   noOverlap=True)
            builder.addOutputTensor(out)

            # Check the shape inference has run.
            assert builder.getTensorShape(out) == list(data.shape)

            result.append(out)
        return result

    def reference(ref_data):
        result = []
        for sliceid in range(4):
            result.append(data[:, sliceid * 3:(sliceid + 1) * 3, :])
        return result

    op_tester.setPatterns(popart.PatternsLevel.All, enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'infer')


# Test training of non-overlapping dynamic slices
# Each slice is scaled by a different factor to produce varying gradients
def test_dynamicslice_training(op_tester):
    data = np.random.rand(5, 12, 7).astype(np.float32)
    axes = [1]
    sizes = [3]

    def init_builder(builder):
        tensor = builder.addInitializedInputTensor(data)
        outputs = []
        result = []
        for sliceid in range(4):
            index = builder.addInputTensor(np.asarray([sliceid * 3],
                                                      np.uint32))
            out = builder.aiGraphcore.dynamicslice([tensor, index],
                                                   axes=axes,
                                                   sizes=sizes,
                                                   noOverlap=True)
            out = builder.aiGraphcore.scale([out], float(1 + sliceid))

            # Check the shape inference has run.
            assert builder.getTensorShape(out) == list(data.shape)

            outputs.append(out)
            result.append(out)

        sum = builder.aiOnnx.sum(outputs)
        sum = builder.aiOnnx.reducesum([sum], axes=[0, 1, 2], keepdims=False)
        sum = builder.aiOnnx.unsqueeze([sum], axes=[0])

        builder.addOutputTensor(sum)

        result = [
            sum,
            popart.reservedGradientPrefix() + sum,
            popart.reservedGradientPrefix() + tensor
        ] + result
        return result

    def reference(ref_data):
        tensor = torch.tensor(data, requires_grad=True)
        outputs = []
        result = []
        for sliceid in range(4):
            out = tensor[:, sliceid * 3:(sliceid + 1) * 3, :]
            out = out * float(1 + sliceid)
            outputs.append(out)
            result.append(out)

        sum = torch.unsqueeze(torch.sum(torch.stack(outputs)), dim=0)

        d__o = ref_data.getOutputTensorGrad(0)
        sum.backward(torch.tensor(d__o))

        result = [sum, torch.tensor(d__o), tensor.grad] + result
        return result

    op_tester.setPatterns(popart.PatternsLevel.All, enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'train')


# Test to show that the gradient of dynamic slices are incorrect if noOverlap is
# set to True while the sliced regions overlap
def test_dynamicslice_overlap_wrong(op_tester):
    data = np.random.rand(10).astype(np.float32)
    axes = [0]
    sizes = [5]

    def init_builder(builder):
        tensor = builder.addInitializedInputTensor(data)
        outputs = []
        result = []
        for sliceid in range(2):
            index = builder.addInputTensor(np.asarray([sliceid * 3],
                                                      np.uint32))
            out = builder.aiGraphcore.dynamicslice([tensor, index],
                                                   axes=axes,
                                                   sizes=sizes,
                                                   noOverlap=True)
            out = builder.aiGraphcore.scale([out], float(1 + sliceid))

            # Check the shape inference has run.
            assert builder.getTensorShape(out) == list(data.shape)

            outputs.append(out)
            result.append(out)

        sum = builder.aiOnnx.sum(outputs)
        sum = builder.aiOnnx.reducesum([sum], axes=[0], keepdims=False)
        sum = builder.aiOnnx.unsqueeze([sum], axes=[0])

        builder.addOutputTensor(sum)
        result = [
            sum,
            popart.reservedGradientPrefix() + sum,
            popart.reservedGradientPrefix() + tensor
        ] + result
        return result

    def reference(ref_data):
        tensor = torch.tensor(data, requires_grad=True)
        outputs = []
        result = []
        for sliceid in range(2):
            out = tensor[sliceid * 3:sliceid * 3 + sizes[0]]
            out = out * float(1 + sliceid)
            outputs.append(out)
            result.append(out)

        sum = torch.unsqueeze(torch.sum(torch.stack(outputs)), dim=0)

        d__o = ref_data.getOutputTensorGrad(0)
        sum.backward(torch.tensor(d__o))

        # Note: We have to adjust the values here to make the comparison equal,
        # but dynamicslice with noOverlap=True gives a wrong gradient result
        # due to overlapping slices
        tensor.grad += torch.tensor(
            np.asarray([0.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

        result = [sum, torch.tensor(d__o), tensor.grad] + result
        return result

    op_tester.setPatterns(popart.PatternsLevel.All, enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'train')


# Test to show that the gradient of dynamic slices are correct if noOverlap is
# set to False while the sliced regions overlap
def test_dynamicslice_overlap_correct(op_tester):
    data = np.random.rand(10).astype(np.float32)
    axes = [0]
    sizes = [5]

    def init_builder(builder):
        tensor = builder.addInitializedInputTensor(data)
        outputs = []
        result = []
        for sliceid in range(2):
            index = builder.addInputTensor(np.asarray([sliceid * 3],
                                                      np.uint32))
            out = builder.aiGraphcore.dynamicslice([tensor, index],
                                                   axes=axes,
                                                   sizes=sizes,
                                                   noOverlap=False)
            out = builder.aiGraphcore.scale([out], float(1 + sliceid))

            # Check the shape inference has run.
            assert builder.getTensorShape(out) == list(data.shape)

            outputs.append(out)
            result.append(out)

        sum = builder.aiOnnx.sum(outputs)
        sum = builder.aiOnnx.reducesum([sum], axes=[0], keepdims=False)
        sum = builder.aiOnnx.unsqueeze([sum], axes=[0])

        builder.addOutputTensor(sum)
        result = [
            sum,
            popart.reservedGradientPrefix() + sum,
            popart.reservedGradientPrefix() + tensor
        ] + result
        return result

    def reference(ref_data):
        tensor = torch.tensor(data, requires_grad=True)
        outputs = []
        result = []
        for sliceid in range(2):
            out = tensor[sliceid * 3:sliceid * 3 + sizes[0]]
            out = out * float(1 + sliceid)
            outputs.append(out)
            result.append(out)

        sum = torch.unsqueeze(torch.sum(torch.stack(outputs)), dim=0)

        d__o = ref_data.getOutputTensorGrad(0)
        sum.backward(torch.tensor(d__o))

        # Note: Comparison equal with noOverlap=False handles overlapping
        # slices correctly (at higher computational cost). No correction needed.
        tensor.grad += torch.tensor(
            np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

        result = [sum, torch.tensor(d__o), tensor.grad] + result
        return result

    op_tester.setPatterns(popart.PatternsLevel.All, enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'train')


# Tests the situation in which there is no add or sum for the dynamic slice
# gradient
def test_non_sum_add_grad_op(op_tester):
    data = np.random.rand(4).astype(np.float32)

    def init_builder(builder):
        tensor = builder.addInitializedInputTensor(data)
        start = builder.addInputTensor(np.asarray(0).astype(np.uint32))

        bias = builder.addInitializedInputTensor(
            np.asarray(1).astype(np.float32))

        shape_c = builder.aiOnnx.constant(np.array([2, 2]).astype(np.int64))

        out = builder.aiOnnx.add([tensor, bias])

        out = builder.aiOnnx.reshape([out, shape_c])
        out = builder.aiGraphcore.dynamicslice([out, start],
                                               axes=[0],
                                               sizes=[2],
                                               noOverlap=True)

        sum = builder.aiOnnx.reducesum([out], axes=[0, 1], keepdims=False)

        builder.addOutputTensor(sum)

        result = [
            sum,
            popart.reservedGradientPrefix() + sum,
            popart.reservedGradientPrefix() + tensor
        ]
        return result

    def reference(ref_data):
        tensor = torch.tensor(data, requires_grad=True)
        print("tensor")
        print(tensor)
        bias = torch.nn.Parameter(torch.tensor([1], dtype=torch.float32))

        out = tensor + bias
        out = out.reshape((2, 2))
        print("reshaped")
        print(out)

        out = out[0:2]

        print("out")
        print(out)
        out = torch.sum(out)

        d__o = ref_data.getOutputTensorGrad(0)
        out.backward(torch.tensor(d__o))

        result = [out, torch.tensor(d__o), tensor.grad]
        return result

    op_tester.setPatterns(popart.PatternsLevel.All, enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'train')
