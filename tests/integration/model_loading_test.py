# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
import pytest


def test_bad_model_proto():
    builder = popart.Builder()

    i = builder.addInputTensor("FLOAT", [4, 4])
    c = builder.aiOnnx.constant(np.random.rand(4, 4).astype(np.float32))
    m = builder.aiOnnx.matmul([i, c])
    m = builder.aiOnnx.matmul([m, c])

    builder.addOutputTensor(m)

    proto = builder.getModelProto()

    # Invalidate the onnx model proto
    proto = [i for i in proto]
    i = 0
    while i < len(proto):
        _ = proto[i]
        proto[i] = 10
        i += 10
    proto = bytes(proto)

    # Attempt to load the model proto
    with pytest.raises(popart.popart_exception) as e_info:
        builder = popart.Builder(proto)

    assert (e_info.value.args[0].startswith(
        'Failed to load a ModelProto from the string '))
    assert (e_info.value.args[0].endswith(
        'Check that it is either a valid path to an existing onnx'
        ' model file, or is a valid onnx ModelProto string.'))
