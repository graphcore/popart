import pytest

import poponnx

def test_basic():

    builder = poponnx.Builder()

    i1 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))
    i2 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1, 2, 32, 32]))

    o = builder.add(i1, i2)

    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    assert(len(proto) > 0)
    assert(len(i1) > 0)
    assert(len(i2) > 0)
    assert(len(o) > 0)
    assert(i1 != i2)
    assert(i2 != o)
