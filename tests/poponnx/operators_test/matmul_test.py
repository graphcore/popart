import numpy as np
import poponnx
import torch
import pytest
from op_tester import op_tester

# `import test_util` requires adding to sys.path
import sys
from pathlib import Path
sys.path.append(Path(__file__).resolve().parent.parent)
import test_util as tu


def test_matmul(op_tester):
    d1 = np.random.rand(2, 3).astype(np.float32)
    d2 = np.random.rand(3, 4).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        o = builder.aiOnnx.matmul([i1, i2])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        out = np.matmul(d1, d2)
        return [out]

    op_tester.run(init_builder, reference)


def test_matmul_mismatched_inputs(op_tester):
    """
    Test the exception raised when the inputs to a matmul are mismatched
    """

    d1 = np.random.rand(3, 4).astype(np.float32)
    d2 = np.random.rand(3, 4).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        o = builder.aiOnnx.matmul([i1, i2])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        out = np.matmul(d1, d2)
        return [out]

    with pytest.raises(poponnx.poponnx_exception) as e_info:
        op_tester.run(init_builder, reference)

    assert (
        e_info.value.args[0] ==
        "Op(ai.onnx.MatMul:9, outputs=[MatMul:0]) contracting dimensions unequal: lhs 'input' [3 4], rhs 'input/1' [3 4]"
    )


def test_matmul_scalar_input(op_tester):
    """
    Test the exception raised when an intput to a matmul is a scalar
    """

    d1 = np.array((2), dtype=np.float32)
    d2 = np.random.rand(3, 4).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        o = builder.aiOnnx.matmul([i1, i2])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        out = np.matmul(d1, d2)
        return [out]

    with pytest.raises(poponnx.poponnx_exception) as e_info:
        op_tester.run(init_builder, reference)

    assert (
        e_info.value.args[0] ==
        "Op(ai.onnx.MatMul:9, outputs=[MatMul:0]) doesn't support scalar tensor input as the lhs input"
    )


def test_matmul_grouped_1(op_tester):
    d1 = np.random.rand(2, 1, 4, 5, 1, 7, 8).astype(np.float32)
    d2 = np.random.rand(2, 3, 1, 5, 6, 8, 9).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        o = builder.aiOnnx.matmul([i1, i2])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        out = np.matmul(d1, d2)
        return [out]

    op_tester.run(init_builder, reference)


def test_matmul_grouped_2(op_tester):
    d1 = np.random.rand(2, 1, 4, 5, 1, 7, 8).astype(np.float32)
    d2 = np.random.rand(2, 3, 4, 5, 6, 8, 9).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        o = builder.aiOnnx.matmul([i1, i2])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        out = np.matmul(d1, d2)
        return [out]

    op_tester.run(init_builder, reference)


def test_matmul_grouped_3(op_tester):
    d1 = np.random.rand(4, 5, 1, 7, 8).astype(np.float32)
    d2 = np.random.rand(2, 3, 1, 5, 6, 8, 9).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        o = builder.aiOnnx.matmul([i1, i2])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        out = np.matmul(d1, d2)
        return [out]

    op_tester.run(init_builder, reference)


def test_matmul_grouped_4(op_tester):
    d1 = np.random.rand(2, 1, 4, 5, 1, 7, 8).astype(np.float32)
    d2 = np.random.rand(4, 5, 6, 8, 9).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        o = builder.aiOnnx.matmul([i1, i2])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        out = np.matmul(d1, d2)
        return [out]

    op_tester.run(init_builder, reference)


def test_matmul_grouped_5(op_tester):
    d1 = np.random.rand(3, 3, 3).astype(np.float32)
    d2 = np.random.rand(3, 3, 4).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        o = builder.aiOnnx.matmul([i1, i2])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        out = np.matmul(d1, d2)
        return [out]

    op_tester.run(init_builder, reference)


def test_matmul_broadcasting(op_tester):
    # generated test cases
    # these are all known to be valid input shapes to np.matmul
    shapes = (
        ([1, 3, 4], [1, 3, 4, 2]),
        ([2, 1, 4], [1, 4, 3]),
        ([3], [3]),
        ([2, 3, 1, 4], [1, 4, 3]),
        ([2, 4, 1, 3], [2, 4, 3, 1]),
        ([3, 1, 2], [2, 4]),
        ([1, 2], [4, 3, 2, 1]),
        ([2, 1, 4, 3], [1, 2, 3, 4]),
        ([3, 1, 2, 4], [3, 4, 2]),
        ([1, 4, 2], [2]),
        ([1, 3], [4, 2, 3, 1]),
        ([4], [4, 2]),
        ([3, 1], [2, 3, 1, 4]),
        ([2], [2]),
        ([3, 2], [2]),
        ([2, 1], [3, 4, 1, 2]),
        ([1, 3], [1, 2, 3, 4]),
        ([3, 4], [3, 4, 1]),
        ([1, 2], [2, 1]),
        ([2], [2, 1]),
        ([2, 4], [4]),
        ([2], [4, 2, 1]),
        ([2, 4, 3], [4, 1, 3, 2]),
        ([1, 4, 3], [3, 1]),
        ([3, 4, 1], [1, 4]),
        ([4], [2, 4, 1]),
        ([1], [3, 1, 2]),
        ([1, 4], [4]),
        ([1, 2, 3, 4], [3, 2, 4, 1]),
        ([3], [1, 3, 4]),
        ([4, 2], [2]),
        ([2, 3], [3, 4]),
        ([3, 4, 1], [1]),
        ([1], [4, 2, 1, 3]),
        ([4, 1], [2, 1, 4]),
        ([1, 4, 3, 2], [2]),
        ([1, 4], [4, 1]),
        ([2, 1, 3, 4], [4, 1]),
        ([4, 3], [3, 4]),
        ([1, 3, 2, 4], [4]),
        ([3, 1, 2, 4], [1, 4, 3]),
        ([2, 3], [1, 3, 2]),
        ([3], [3, 2]),
        ([1], [2, 3, 1, 4]),
        ([2], [4, 2, 3]),
        ([4, 2, 1], [4, 1, 3]),
        ([2, 3], [4, 3, 2]),
        ([2, 3, 1, 4], [4]),
        ([4, 2, 3], [3]),
        ([4, 2, 3], [3, 4]),
        ([1], [2, 1, 4]),
        ([2], [3, 4, 2, 1]),
        ([3, 2, 1, 4], [4, 2]),
        ([3, 2], [4, 3, 2, 1]),
        ([4, 1, 3], [3, 4]),
        ([1, 4], [1, 4, 2]),
        ([4], [4]),
        ([2, 3, 4], [2, 1, 4, 3]),
        ([2, 3, 4, 1], [1]),
        ([2, 1, 4, 3], [3, 4]),
        ([3, 2, 1, 4], [4]),
        ([1, 3], [3, 1]),
        ([1], [1, 4]),
        ([4, 2, 3, 1], [1]),
        ([3, 1, 4, 2], [3, 2, 1]),
        ([2], [1, 2, 3]),
        ([3, 1, 2, 4], [4, 2]),
        ([2, 4], [4, 2]),
        ([2, 1, 3, 4], [3, 4, 2]),
        ([2, 1, 4, 3], [1, 3, 2]),
        ([1], [1]),
        ([1, 3, 4], [4]),
        ([2, 1], [1, 4]),
        ([2, 1, 3], [4, 2, 3, 1]),
        ([3], [2, 3, 1]),
        ([3, 4, 2, 1], [1, 4]),
        ([3, 4], [4, 2]),
        ([4, 1, 3, 2], [3, 2, 1]),
        ([2], [3, 2, 4]),
        ([4], [1, 4, 3]),
        ([4, 2], [1, 4, 2, 3]),
        ([1], [2, 4, 1, 3]),
        ([4, 3, 1], [1]),
        ([3, 4], [1, 4, 3]),
        ([2, 4], [2, 4, 3]),
        ([2], [4, 1, 2, 3]),
        ([2, 1, 4, 3], [3]),
        ([2, 1, 3], [2, 3, 1]),
        ([3, 1, 4, 2], [4, 2, 1]),
        ([3, 4], [1, 4, 2]),
        ([4], [3, 4, 1]),
        ([3], [1, 4, 3, 2]),
        ([1, 3], [1, 3, 2]),
        ([1, 3], [2, 4, 3, 1]),
        ([4, 3, 1], [1, 2]),
        ([3, 2, 4], [4]),
        ([4, 1, 3], [3]),
        ([3], [2, 1, 3, 4]),
        ([3, 1], [1, 2]),
        ([4, 1], [1, 2]),
    )

    def zeros(*args):
        return np.zeros(args, dtype=np.float32)

    for lhs, rhs in shapes:

        d1 = np.random.rand(*lhs).astype(np.float32)
        d2 = np.random.rand(*rhs).astype(np.float32)

        def init_builder(builder):
            i1 = builder.addInputTensor(d1)
            i2 = builder.addInputTensor(d2)
            t1 = builder.aiOnnx.matmul([i1, i2])

            # loss can't handle scalar value produced by `matmul` of 2 1d tensors
            # so include an `add` operation, and put the output of matmul in anchors
            if np.matmul(d1, d2).shape == ():
                i3 = builder.addInputTensor(zeros(2))
                o = builder.aiOnnx.add([i3, t1])
                builder.addOutputTensor(o)
                return [o, t1]
            else:
                builder.addOutputTensor(t1)
                return [t1]

        def reference(ref_data):
            t1 = np.matmul(d1, d2)
            if t1.shape == ():
                out = zeros(2) + t1
                return [out, np.array(t1)]
            else:
                return [t1]

        op_tester.run(init_builder, reference)
