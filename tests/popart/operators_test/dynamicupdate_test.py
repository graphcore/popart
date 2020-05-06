# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import itertools
import numpy as np
import popart
import torch
import pytest
import torch.nn.functional as F
from op_tester import op_tester


# Test a chain of non-overlapping dynamic updates
# init -> U0 -> out0 -> U1 -> out1 -> U2 -> out2
#         ^             ^             ^
#         |             |             |
#         tensor0       tensor1       tensor2
# where tensor0, tensor1 and tensor2 are non-overlapping subregions
# of the out tensor
def test_dynamicupdate(op_tester):
    data0 = np.random.rand(5, 4, 7).astype(np.float32)
    data1 = np.random.rand(5, 4, 7).astype(np.float32)
    data2 = np.random.rand(5, 4, 7).astype(np.float32)
    axes = [1]
    sizes = [4]

    def init_builder(builder):
        tensor0 = builder.addInputTensor(data0)
        tensor1 = builder.addInputTensor(data1)
        tensor2 = builder.addInputTensor(data2)
        tensors = [tensor0, tensor1, tensor2]
        result = []
        out = builder.aiGraphcore.init([5, 12, 7], popart.DataType.FLOAT,
                                       popart.InitType.NoInit, "test_init")
        for sliceid in range(3):
            index = builder.addInputTensor(np.asarray([sliceid * 4],
                                                      np.uint32))
            out = builder.aiGraphcore.dynamicupdate(
                [out, index, tensors[sliceid]],
                axes=axes,
                sizes=sizes,
                noOverlap=True)
            builder.addOutputTensor(out)
        result.append(out)
        return result

    def reference(ref_data):
        result = []
        result.append(np.concatenate((data0, data1, data2), axis=1))
        return result

    op_tester.patterns = popart.PatternsLevel.All
    op_tester.run(init_builder, reference, 'infer')


# Test training of non-overlapping dynamic updates
def test_dynamicupdate_training(op_tester):
    data0 = np.random.rand(5, 4, 7).astype(np.float32)
    data1 = np.random.rand(5, 4, 7).astype(np.float32)
    data2 = np.random.rand(5, 4, 7).astype(np.float32)
    axes = [1]
    sizes = [4]

    def init_builder(builder):
        tensor0 = builder.addInitializedInputTensor(data0)
        tensor1 = builder.addInitializedInputTensor(data1)
        tensor2 = builder.addInitializedInputTensor(data2)
        tensors = [tensor0, tensor1, tensor2]
        result = []
        out = builder.aiGraphcore.init([5, 12, 7], popart.DataType.FLOAT,
                                       popart.InitType.NoInit, "test_init")
        for sliceid in range(3):
            index = builder.addInputTensor(np.asarray([sliceid * 4],
                                                      np.uint32))
            out = builder.aiGraphcore.dynamicupdate(
                [out, index, tensors[sliceid]],
                axes=axes,
                sizes=sizes,
                noOverlap=True)
        result.append(out)

        sum = builder.aiOnnx.reducesum([out], axes=[0, 1, 2], keepdims=False)
        sum = builder.aiOnnx.unsqueeze([sum], axes=[0])

        builder.addOutputTensor(sum)
        result = [
            sum,
            popart.reservedGradientPrefix() + sum,
            popart.reservedGradientPrefix() + tensor0,
            popart.reservedGradientPrefix() + tensor1,
            popart.reservedGradientPrefix() + tensor2,
        ] + result
        return result

    def reference(ref_data):
        tensor0 = torch.tensor(data0, requires_grad=True)
        tensor1 = torch.tensor(data1, requires_grad=True)
        tensor2 = torch.tensor(data2, requires_grad=True)

        outputs = []
        result = []
        out = torch.cat((tensor0, tensor1, tensor2), dim=1)
        outputs.append(out)
        result.append(out)

        sum = torch.unsqueeze(torch.sum(torch.stack(outputs)), dim=0)

        d__o = ref_data.getOutputTensorGrad(0)
        sum.backward(torch.tensor(d__o))

        result = [
            sum,
            torch.tensor(d__o), tensor0.grad, tensor1.grad, tensor2.grad
        ] + result
        return result

    op_tester.patterns = popart.PatternsLevel.All
    op_tester.run(init_builder, reference, 'train')


# Test to show that the gradient of dynamic updates are incorrect if noOverlap
# is set to True while the sliced regions overlap
def test_dynamicupdate_overlap_wrong(op_tester):
    data0 = np.random.rand(5).astype(np.float32)
    data1 = np.random.rand(5).astype(np.float32)
    axes = [0]
    sizes = [5]

    def init_builder(builder):
        tensor0 = builder.addInitializedInputTensor(data0)
        tensor1 = builder.addInitializedInputTensor(data1)
        tensors = [tensor0, tensor1]
        result = []
        out = builder.aiGraphcore.init([10], popart.DataType.FLOAT,
                                       popart.InitType.NoInit, "test_init")
        for sliceid in range(2):
            index = builder.addInputTensor(np.asarray([sliceid * 4],
                                                      np.uint32))
            scaled = builder.aiGraphcore.scale([tensors[sliceid]],
                                               float(1 + sliceid))
            out = builder.aiGraphcore.dynamicupdate([out, index, scaled],
                                                    axes=axes,
                                                    sizes=sizes,
                                                    noOverlap=True)
        result.append(out)

        sum = builder.aiOnnx.reducesum([out], axes=[0], keepdims=False)
        sum = builder.aiOnnx.unsqueeze([sum], axes=[0])

        builder.addOutputTensor(sum)
        result = [
            sum,
            popart.reservedGradientPrefix() + sum,
            popart.reservedGradientPrefix() + tensor0,
            popart.reservedGradientPrefix() + tensor1,
        ] + result
        return result

    def reference(ref_data):
        tensor0 = torch.tensor(data0, requires_grad=True)
        tensor1 = torch.tensor(data1, requires_grad=True)

        outputs = []
        result = []

        out = torch.zeros(10)

        out[0:5] = tensor0 * 1.0
        out[4:9] = tensor1 * 2.0

        outputs.append(out)
        result.append(out)

        sum = torch.unsqueeze(torch.sum(torch.stack(outputs)), dim=0)

        d__o = ref_data.getOutputTensorGrad(0)
        sum.backward(torch.tensor(d__o))

        # Note: We have to adjust the value here to make the comparison equal,
        # but dynamicupdate with noOverlap=True gives a wrong gradient result
        # due to overlapping updates
        tensor0.grad[4] += 0.1

        result = [sum, torch.tensor(d__o), tensor0.grad, tensor1.grad] + result
        return result

    op_tester.patterns = popart.PatternsLevel.All
    op_tester.run(init_builder, reference, 'train')


# Test to show that the gradient of dynamic updates are correct if noOverlap is
# set to False while the sliced regions overlap
def test_dynamicupdate_overlap_correct(op_tester):
    data0 = np.random.rand(5).astype(np.float32)
    data1 = np.random.rand(5).astype(np.float32)
    axes = [0]
    sizes = [5]

    def init_builder(builder):
        tensor0 = builder.addInitializedInputTensor(data0)
        tensor1 = builder.addInitializedInputTensor(data1)
        tensors = [tensor0, tensor1]
        result = []
        out = builder.aiGraphcore.init([10], popart.DataType.FLOAT,
                                       popart.InitType.NoInit, "test_init")
        for sliceid in range(2):
            index = builder.addInputTensor(np.asarray([sliceid * 4],
                                                      np.uint32))
            scaled = builder.aiGraphcore.scale([tensors[sliceid]],
                                               float(1 + sliceid))

            out = builder.aiGraphcore.dynamicupdate([out, index, scaled],
                                                    axes=axes,
                                                    sizes=sizes,
                                                    noOverlap=False)

        result.append(out)

        sum = builder.aiOnnx.reducesum([out], axes=[0], keepdims=False)
        sum = builder.aiOnnx.unsqueeze([sum], axes=[0])

        builder.addOutputTensor(sum)
        result = [
            sum,
            popart.reservedGradientPrefix() + sum,
            popart.reservedGradientPrefix() + tensor0,
            popart.reservedGradientPrefix() + tensor1,
        ] + result
        return result

    def reference(ref_data):
        tensor0 = torch.tensor(data0, requires_grad=True)
        tensor1 = torch.tensor(data1, requires_grad=True)

        outputs = []
        result = []

        out = torch.zeros(10)

        out[0:5] = tensor0 * 1.0
        out[4:9] = tensor1 * 2.0

        outputs.append(out)
        result.append(out)

        sum = torch.unsqueeze(torch.sum(torch.stack(outputs)), dim=0)

        d__o = ref_data.getOutputTensorGrad(0)
        sum.backward(torch.tensor(d__o))

        # Note: Comparison equal with noOverlap=False handles overlapping
        # updates correctly (at higher computational cost).
        # No correction needed.
        tensor0.grad[4] += 0.0

        result = [sum, torch.tensor(d__o), tensor0.grad, tensor1.grad] + result
        return result

    op_tester.patterns = popart.PatternsLevel.All
    op_tester.run(init_builder, reference, 'train')
