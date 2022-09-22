# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
import pytest
import test_util as tu


@tu.requires_ipu_model
def test_constant_weights_write_error():
    """
    Create an InferenceSession, try to use writeWeights, and see that an
    error is thrown.
    """
    opts = popart.SessionOptions()
    builder = popart.Builder()
    input = builder.addInputTensor(popart.TensorInfo("FLOAT", [1]))
    weight = builder.addInitializedInputTensor(np.array([10], dtype=np.float32))
    output = builder.aiOnnx.add([input, weight])
    builder.addOutputTensor(output)

    with tu.create_test_device(1) as device:
        session = popart.InferenceSession(
            builder.getModelProto(),
            dataFlow=popart.DataFlow(1, {output: popart.AnchorReturnType("ALL")}),
            deviceInfo=device,
            userOptions=opts,
        )
        session.prepareDevice()
        pyWeight = np.array([100], dtype=np.float32)
        with pytest.raises(popart.popart_exception) as e_info:
            session.writeWeights(popart.PyWeightsIO({weight: pyWeight}))

    assert e_info.value.args[0].endswith(
        "Cannot call writeWeights when constantWeights is "
        "set. Set `constantWeights` to False"
        " in the `SessionOption`s when initialising the"
        " session to enable this behaviour."
    )
