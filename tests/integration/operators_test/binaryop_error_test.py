# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart
import pytest


def test_binaryop_error():
    M = 10
    N = 10
    K = 5
    builder = popart.Builder(opsets={"ai.onnx": 9, "ai.graphcore": 1})

    seq_info_A = popart.TensorInfo("FLOAT", [M, N])
    A = builder.addInputTensor(seq_info_A, "A")
    seq_info_B = popart.TensorInfo("FLOAT", [N, K])
    B = builder.addInputTensor(seq_info_B, "B")

    C = builder.aiOnnx.add([A, B], "C")
    outputs = {C: popart.AnchorReturnType("ALL")}
    for k in outputs.keys():
        builder.addOutputTensor(k)

    data_flow = popart.DataFlow(1, outputs)
    model_opts = {"numIPUs": 1}
    device = popart.DeviceManager().createIpuModelDevice(model_opts)
    proto = builder.getModelProto()

    with pytest.raises(popart.popart_exception) as e_info:
        _ = popart.InferenceSession(
            fnModel=proto, deviceInfo=device, dataFlow=data_flow
        )
    assert "np broadcasting failed on 'Op" in e_info.value.args[0]
    assert "ai.onnx.Add" in e_info.value.args[0]
    assert "frames [10, 10] and [10, 5] are not aligned" in e_info.value.args[0]
