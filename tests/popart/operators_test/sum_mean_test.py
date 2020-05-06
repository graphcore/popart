# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
import torch
import pytest
import torch.nn.functional as F
from op_tester import op_tester

# `import test_util` requires adding to sys.path
import sys
from pathlib import Path
sys.path.append(Path(__file__).resolve().parent.parent)

import test_util as tu


def test_sum(op_tester):
    d1 = np.random.rand(1, 2, 1).astype(np.float32)
    d2 = np.random.rand(1, 1, 2).astype(np.float32)
    d3 = np.random.rand(2, 1, 1).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        i3 = builder.addInputTensor(d3)
        o = builder.aiOnnx.sum([i1, i2, i3], "test_sum")
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        out = d1 + d2 + d3
        return [out]

    op_tester.patterns = ['PreUniRepl']
    op_tester.run(init_builder, reference, 'infer')


def test_sum_1_input(op_tester):
    d1 = np.random.rand(2, 2).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.sum([i1], "test_sum")
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        return [d1]

    op_tester.patterns = []
    op_tester.run(init_builder, reference, 'infer')


def test_X_training(op_tester, X="mean"):
    d1 = np.random.rand(8, 7).astype(np.float32)
    d2 = np.random.rand(6, 8, 7).astype(np.float32)
    d3 = np.random.rand(6, 1, 7).astype(np.float32)
    d4 = np.random.rand(7).astype(np.float32)
    d5 = np.random.rand(6, 8, 7).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        i3 = builder.addInputTensor(d3)
        i4 = builder.addInputTensor(d4)
        i5 = builder.addInputTensor(d5)
        if X == "mean":
            o = builder.aiOnnx.mean([i1, i2, i3, i4, i5], "test_%s" % (X, ))
        elif X == "sum":
            o = builder.aiOnnx.sum([i1, i2, i3, i4, i5], "test_%s" % (X, ))
        else:
            raise RuntimeError(
                "Unexpected type X in test_X_training, should be mean or sum")

        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + i1,
            popart.reservedGradientPrefix() + i2,
            popart.reservedGradientPrefix() + i3,
            popart.reservedGradientPrefix() + i4,
            popart.reservedGradientPrefix() + i5,
            popart.reservedGradientPrefix() + o
        ]

    def reference(ref_data):
        t1 = torch.tensor(d1, requires_grad=True)
        t2 = torch.tensor(d2, requires_grad=True)
        t3 = torch.tensor(d3, requires_grad=True)
        t4 = torch.tensor(d4, requires_grad=True)
        t5 = torch.tensor(d5, requires_grad=True)

        out = torch.add(t1, t2)
        out = torch.add(t3, out)
        out = torch.add(t4, out)
        out = torch.add(t5, out)

        if X == "mean":
            out = torch.div(out, 5)
        elif X == "sum":
            pass
        else:
            raise RuntimeError(
                "Unexpected type X in test_X_training, should be mean or sum")

        d__o = ref_data.getOutputTensorGrad(0)
        out.backward(torch.tensor(d__o))

        return [out, t1.grad, t2.grad, t3.grad, t4.grad, t5.grad, d__o]

    op_tester.patterns = ['OpToIdentity']
    op_tester.run(init_builder, reference, 'train')


def test_mean_training(op_tester):
    test_X_training(op_tester, "mean")


def test_sum_training(op_tester):
    test_X_training(op_tester, "sum")


# Test with opset 6 - all have to be the same type and shape
def test_mean_training_2(op_tester):
    d1 = np.random.rand(8, 7).astype(np.float32)
    d2 = np.random.rand(8, 7).astype(np.float32)
    d3 = np.random.rand(8, 7).astype(np.float32)
    d4 = np.random.rand(8, 7).astype(np.float32)
    d5 = np.random.rand(8, 7).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        i3 = builder.addInputTensor(d3)
        i4 = builder.addInputTensor(d4)
        i5 = builder.addInputTensor(d5)
        o = builder.aiOnnx.mean([i1, i2, i3, i4, i5], "test_max")
        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + i1,
            popart.reservedGradientPrefix() + i2,
            popart.reservedGradientPrefix() + i3,
            popart.reservedGradientPrefix() + i4,
            popart.reservedGradientPrefix() + i5,
            popart.reservedGradientPrefix() + o
        ]

    def reference(ref_data):
        t1 = torch.tensor(d1, requires_grad=True)
        t2 = torch.tensor(d2, requires_grad=True)
        t3 = torch.tensor(d3, requires_grad=True)
        t4 = torch.tensor(d4, requires_grad=True)
        t5 = torch.tensor(d5, requires_grad=True)

        out = torch.add(t1, t2)
        out = torch.add(t3, out)
        out = torch.add(t4, out)
        out = torch.add(t5, out)

        out = torch.div(out, 5)

        d__o = ref_data.getOutputTensorGrad(0)
        out.backward(torch.tensor(d__o))

        return [out, t1.grad, t2.grad, t3.grad, t4.grad, t5.grad, d__o]

    op_tester.patterns = ['OpToIdentity']
    op_tester.run(init_builder, reference, 'train', {
        "ai.onnx": 7,
        "ai.graphcore": 1
    })


# Test with opset 6 with incorrect inputs i.e. different shapes
def test_mean_training_3(op_tester):
    d1 = np.random.rand(8, 7).astype(np.float32)
    d2 = np.random.rand(6, 8, 7).astype(np.float32)
    d3 = np.random.rand(6, 1, 7).astype(np.float32)
    d4 = np.random.rand(7).astype(np.float32)
    d5 = np.random.rand(6, 8, 7).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        i3 = builder.addInputTensor(d3)
        i4 = builder.addInputTensor(d4)
        i5 = builder.addInputTensor(d5)
        o = builder.aiOnnx.mean([i1, i2, i3, i4, i5], "test_max")
        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + i1,
            popart.reservedGradientPrefix() + i2,
            popart.reservedGradientPrefix() + i3,
            popart.reservedGradientPrefix() + i4,
            popart.reservedGradientPrefix() + i5,
            popart.reservedGradientPrefix() + o
        ]

    def reference(ref_data):
        t1 = torch.tensor(d1, requires_grad=True)
        t2 = torch.tensor(d2, requires_grad=True)
        t3 = torch.tensor(d3, requires_grad=True)
        t4 = torch.tensor(d4, requires_grad=True)
        t5 = torch.tensor(d5, requires_grad=True)

        out = torch.add(t1, t2)
        out = torch.add(t3, out)
        out = torch.add(t4, out)
        out = torch.add(t5, out)

        out = torch.div(out, 5)

        d__o = ref_data.getOutputTensorGrad(0)
        out.backward(torch.tensor(d__o))

        return [out, t1.grad, t2.grad, t3.grad, t4.grad, t5.grad, d__o]

    op_tester.patterns = ['OpToIdentity']

    with pytest.raises(popart.popart_exception) as e_info:
        op_tester.run(init_builder, reference, 'train', {
            "ai.onnx": 7,
            "ai.graphcore": 1
        })

    assert ("Inputs to ai.onnx.Mean:6 do not all the same type & shape" in
            e_info.value.args[0])
