# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import sys
from pathlib import Path
import numpy as np
import torch

import popart

# `import test_util` requires adding to sys.path
sys.path.append(Path(__file__).resolve().parent.parent)
import test_util as tu


def test_topk_2d(op_tester):
    d1 = np.random.rand(7, 8).astype(np.float32) * 10
    k = 4
    for axis in [0, 1]:

        def init_builder(builder):
            i1 = builder.addInputTensor(d1)
            k_t = builder.aiOnnx.constant(np.array([k]).astype(np.int64))
            [vals, inds] = builder.aiOnnx.topk([i1, k_t], axis=axis)
            builder.addOutputTensor(vals)
            return [vals, inds]

        def reference(_):  # ref_data is an unused argument
            a = torch.tensor(d1)
            b = torch.topk(a, k=k, dim=axis)
            return [b.values, b.indices]

        # Torch doesn't have a uint32 type
        op_tester.check_dtypes = False
        op_tester.run(init_builder, reference, "infer")


def test_topk_2d_smallest(op_tester):
    d1 = np.random.rand(7, 8).astype(np.float32) * 10
    k = 4
    for axis in [0, 1]:

        def init_builder(builder):
            i1 = builder.addInputTensor(d1)
            k_t = builder.aiOnnx.constant(np.array([k]).astype(np.int64))
            [vals, inds] = builder.aiOnnx.topk([i1, k_t], axis=axis, largest=0)

            builder.addOutputTensor(vals)
            return [vals, inds]

        def reference(_):  # ref_data is an unused argument
            a = torch.tensor(d1)
            b = torch.topk(a, k=k, dim=axis, largest=False)
            return [b.values, b.indices]

        # Torch doesn't have a uint32 type
        op_tester.check_dtypes = False
        op_tester.run(
            init_builder, reference, "infer", opsets={"ai.onnx": 11, "ai.graphcore": 1}
        )


def test_topk_2d_sorted():
    np.random.seed(0)
    d1 = np.random.rand(7, 8).astype(np.float32) * 10
    k = 4

    def run_test(sort_topk):
        if sort_topk:
            sort_topk = 1
        else:
            sort_topk = 0

        bld = popart.Builder(opsets={"ai.onnx": 11, "ai.graphcore": 1})
        i0 = bld.addInputTensor(popart.TensorInfo("FLOAT", [7, 8]))
        k_t = bld.aiOnnx.constant(np.array([k]).astype(np.int64))
        [vals, _] = bld.aiOnnx.topk([i0, k_t], axis=0, sorted=sort_topk)

        bld.addOutputTensor(vals)

        with tu.create_test_device() as device:
            sess = popart.InferenceSession(
                bld.getModelProto(),
                deviceInfo=device,
                dataFlow=popart.DataFlow(1, [vals]),
            )

            sess.prepareDevice()
            anchors = sess.initAnchorArrays()
            stepio = popart.PyStepIO({i0: d1}, anchors)
            sess.run(stepio)
        return anchors[vals]

    sorted_output = run_test(True)
    unsorted_output = run_test(False)

    # The values should not be equal, as one should
    # be sorted and the other should not.
    assert not np.allclose(sorted_output, unsorted_output)

    # The sums of the values should be equal, as they should
    # be the same values.
    assert np.isclose(np.sum(sorted_output), np.sum(unsorted_output))


def test_topk_2d_grad(op_tester):
    d1 = np.random.rand(7, 8).astype(np.float32) * 10
    k = 4
    for axis in [0, 1]:

        def init_builder(builder):
            i1 = builder.addInputTensor(d1)
            k_t = builder.aiOnnx.constant(np.array([k]).astype(np.int64))
            [vals, inds] = builder.aiOnnx.topk([i1, k_t], axis=axis)
            builder.addOutputTensor(vals)
            return [
                vals,
                inds,
                popart.reservedGradientPrefix() + i1,
                popart.reservedGradientPrefix() + vals,
            ]

        def reference(ref_data):
            a = torch.tensor(d1, requires_grad=True)
            b = torch.topk(a, k=k, dim=axis)
            d__o = ref_data.getOutputTensorGrad(0)
            b.values.backward(torch.tensor(d__o))
            return [b.values, b.indices, a.grad, None]

        # Torch doesn't have a uint32 type
        op_tester.check_dtypes = False
        op_tester.run(init_builder, reference, "train")


def test_topk_2d_smallest_grad(op_tester):
    d1 = np.random.rand(7, 8).astype(np.float32) * 10
    k = 4
    for axis in [0, 1]:

        def init_builder(builder):
            i1 = builder.addInputTensor(d1)
            k_t = builder.aiOnnx.constant(np.array([k]).astype(np.int64))
            [vals, inds] = builder.aiOnnx.topk([i1, k_t], axis=axis, largest=0)

            builder.addOutputTensor(vals)
            return [
                vals,
                inds,
                popart.reservedGradientPrefix() + i1,
                popart.reservedGradientPrefix() + vals,
            ]

        def reference(ref_data):
            a = torch.tensor(d1, requires_grad=True)
            b = torch.topk(a, k=k, dim=axis, largest=False)
            d__o = ref_data.getOutputTensorGrad(0)
            b.values.backward(torch.tensor(d__o))
            return [b.values, b.indices, a.grad, None]

        # Torch doesn't have a uint32 type
        op_tester.check_dtypes = False
        op_tester.run(
            init_builder, reference, "train", opsets={"ai.onnx": 11, "ai.graphcore": 1}
        )


def test_topk_2d_unsorted_grad(op_tester):
    d1 = np.random.rand(7, 8).astype(np.float32) * 10
    k = 4
    # for axis in [0, 1]:
    for axis in [0]:

        def init_builder(builder):
            i1 = builder.addInputTensor(d1)
            k_t = builder.aiOnnx.constant(np.array([k]).astype(np.int64))
            [vals, inds] = builder.aiOnnx.topk([i1, k_t], axis=axis, sorted=0)
            builder.setAvailableMemoryProportion({vals, inds}, 0.9)

            builder.addOutputTensor(vals)
            return [
                vals,
                inds,
                popart.reservedGradientPrefix() + i1,
                popart.reservedGradientPrefix() + vals,
            ]

        def reference(ref_data):
            a = torch.tensor(d1, requires_grad=True)
            b = torch.topk(a, k=k, dim=axis, sorted=False)
            d__o = ref_data.getOutputTensorGrad(0)
            b.values.backward(torch.tensor(d__o))
            # Not comparing forward pass output.
            # Unsorted topk vals will not be equal.
            return [None, None, a.grad, None]

        # Torch doesn't have a uint32 type
        op_tester.check_dtypes = False
        op_tester.run(
            init_builder, reference, "train", opsets={"ai.onnx": 11, "ai.graphcore": 1}
        )
