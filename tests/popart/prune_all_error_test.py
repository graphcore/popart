# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import pytest
import popart


def test_prune_all_error():
    # Test that when all operations of a graph get pruned by the pruner a user-friendly exception is raised.

    # Build an onnx model that performs an element-wise sprt operation
    # on a 1D tensor containing 2 elements
    builder = popart.Builder()
    in0 = builder.addInputTensor("FLOAT", [2])
    out = builder.aiOnnx.sqrt([in0])

    # Construct a Session that can run this model. The ingredients:

    # the onnx model
    model = builder.getModelProto()

    # the device to execute on (ipu, ipu model, or cpu)
    deviceInfo = popart.DeviceManager().createCpuDevice()

    # the anchors, or which model tensors to return to the host
    anchors = [in0]

    # how many times to execute the model before returning to the host
    batchesPerStep = 1

    # Exception should be thrown here.
    with pytest.raises(popart.popart_exception) as e_info:
        session = popart.InferenceSession(fnModel=model,
                                          deviceInfo=deviceInfo,
                                          dataFlow=popart.DataFlow(
                                              batchesPerStep, anchors))

    assert (e_info.value.args[0] ==
            "All operations in the main graph were pruned, nothing to compute")
