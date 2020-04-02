import numpy as np
import pytest
import popart
from op_tester import op_tester
import json
from typing import List


def grad(dY, X, dt):
    axes = []
    offset = dY.ndim - X.ndim
    for i in range(0, dY.ndim):
        if i < offset or X.shape[i - offset] == 1:
            axes.append(int(i))
    axis_tup = tuple(axis for axis in axes)
    dX = np.sum(a=dY, axis=axis_tup)
    dX = dX.reshape(X.shape)
    return dX


def run_and_test_value(op_tester, inplace, init_builder, reference, mode):
    if inplace:
        op_tester.passes = ['InPlace']
    session = op_tester.run(init_builder, reference, mode, {
        "ai.onnx": 8,
        "ai.graphcore": 1
    })
    if inplace:
        ir = json.loads(session._serializeIr(
            popart.IrSerializationFormat.JSON))
        graph = ir['maingraph']

        inplace = [op for op in graph if op['type'] == 'ExpandInplace']
        assert len(inplace) == 1
        assert inplace[0]['domain'] == 'ai.graphcore'


def expand(op_tester, inplace):
    d1 = np.random.rand(3, 1).astype(np.float32)
    d2 = np.array([2, 1, 6]).astype(np.int64)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        c = builder.aiOnnx.constant(d2)
        o = builder.aiOnnx.expand([i1, c])
        if inplace:
            o_identity = builder.aiOnnx.identity([o])
            builder.addOutputTensor(o_identity)
            return [o_identity]
        else:
            builder.addOutputTensor(o)
            return [
                o,
                popart.reservedGradientPrefix() + o,
                popart.reservedGradientPrefix() + i1
            ]

    def reference(ref_data):
        expanded = d1 * np.ones(d2, dtype=np.float32)
        if inplace:
            return [expanded]
        dY = ref_data.getOutputTensorGrad(0)
        dX = grad(dY, d1, np.float32)
        return [expanded, dY, dX]

    if inplace:  #grad is tested only as part of non invariant version to avoid duplication
        mode = 'infer'
    else:
        mode = 'train'
    run_and_test_value(op_tester, inplace, init_builder, reference, mode)


def expand_unwind(op_tester, inplace):
    d1 = np.random.rand(6).astype(np.float32)
    d2 = np.array([3, 6]).astype(np.int64)
    d3 = np.random.rand(6, 3).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        c = builder.aiOnnx.constant(d2)
        e = builder.aiOnnx.expand([i1, c])
        i2 = builder.addInputTensor(d3)
        m = builder.aiOnnx.matmul([e, i2])

        if inplace:
            m_identity = builder.aiOnnx.identity([m])
        else:
            m_identity = m
        builder.addOutputTensor(m_identity)
        return [m_identity]

    def reference(ref_data):
        expanded = d1 * np.ones(d2, dtype=np.float32)
        m = expanded.dot(d3)
        return [m]

    run_and_test_value(op_tester, inplace, init_builder, reference, 'infer')


def expand_scalar(op_tester, inplace):
    d1 = np.array((3.0), dtype=np.float32)
    d2 = np.array([2, 1, 6]).astype(np.int64)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        c = builder.aiOnnx.constant(d2)
        o = builder.aiOnnx.expand([i1, c])
        if inplace:
            o_identity = builder.aiOnnx.identity([o])
            builder.addOutputTensor(o_identity)
            return [o_identity]
        else:
            builder.addOutputTensor(o)
            return [
                o,
                popart.reservedGradientPrefix() + o,
                popart.reservedGradientPrefix() + i1
            ]

    def reference(ref_data):
        expanded = d1 * np.ones(d2, dtype=np.float32)
        if inplace:
            return [expanded]
        dY = ref_data.getOutputTensorGrad(0)
        dX = grad(dY, d1, np.float32)
        return [expanded, dY, dX]

    if inplace:  #grad is tested only as part of non invariant version to avoid duplication
        mode = 'infer'
    else:
        mode = 'train'
    run_and_test_value(op_tester, inplace, init_builder, reference, mode)


def expand_smaller_output(op_tester, inplace):
    d1 = np.random.rand(3, 1).astype(np.float32)
    d2 = np.array([2, 1, 1]).astype(np.int64)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        c = builder.aiOnnx.constant(d2)
        o = builder.aiOnnx.expand([i1, c])
        if inplace:
            o_identity = builder.aiOnnx.identity([o])
            builder.addOutputTensor(o_identity)
            return [o_identity]
        else:
            builder.addOutputTensor(o)
            return [
                o,
                popart.reservedGradientPrefix() + o,
                popart.reservedGradientPrefix() + i1
            ]

    def reference(ref_data):
        expanded = d1 * np.ones(d2, dtype=np.float32)
        if inplace:
            return [expanded]
        dY = ref_data.getOutputTensorGrad(0)
        dX = grad(dY, d1, np.float32)
        return [expanded, dY, dX]

    if inplace:  #grad is tested only as part of non invariant version to avoid duplication
        mode = 'infer'
    else:
        mode = 'train'
    run_and_test_value(op_tester, inplace, init_builder, reference, mode)


def expand_non_one_smaller_output(op_tester, inplace):
    d1 = np.random.rand(5, 4, 3).astype(np.float32)
    d2 = np.array([3, 4, 2]).astype(np.int64)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        c = builder.aiOnnx.constant(d2)
        o = builder.aiOnnx.expand([i1, c])
        if inplace:
            o_identity = builder.aiOnnx.identity([o])
        else:
            o_identity = o
        builder.addOutputTensor(o_identity)
        return [
            o_identity,
            popart.reservedGradientPrefix() + o,
            popart.reservedGradientPrefix() + i1
        ]

    with pytest.raises(popart.popart_exception) as e_info:
        if inplace:
            op_tester.passes = ['InPlace']
        session = op_tester.run(init_builder, None, 'infer')
        if inplace:
            ir = json.loads(
                session._serializeIr(popart.IrSerializationFormat.JSON))
            graph = ir['maingraph']

            inplace = [op for op in graph if op['type'] == 'ExpandInplace']
            assert len(inplace) == 1
            assert inplace[0]['domain'] == 'ai.graphcore'

    assert (
        'corresponding dimensions must have the same value or one of them must be 1'
    ) in str(e_info.value)


def test_expand(op_tester):
    expand(op_tester, False)


def test_expand_inplace(op_tester):
    expand(op_tester, True)


def test_expand_smaller_output(op_tester):
    expand_smaller_output(op_tester, False)


def test_expand_smaller_output_inplace(op_tester):
    expand_smaller_output(op_tester, True)


def test_expand_non_one_smaller_output(op_tester):
    expand_non_one_smaller_output(op_tester, False)


def test_expand_non_one_smaller_output_inplace(op_tester):
    expand_non_one_smaller_output(op_tester, True)


def test_expand_scalar(op_tester):
    expand_scalar(op_tester, False)


def test_expand_scalar_inplace(op_tester):
    expand_scalar(op_tester, True)


def test_expand_unwind(op_tester):
    expand_unwind(op_tester, False)
