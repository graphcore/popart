# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import popart
import pytest
import test_util as tu


# Both manual and auto recomputation are not supported
# concurrently
def test_valid_recompute_options():
    builder = popart.Builder()

    i1 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1]))
    i2 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1]))
    o = builder.aiOnnx.add([i1, i2])

    builder.addOutputTensor(o)

    # specify manual recomputation
    builder.recomputeOutputInBackwardPass(o)

    # specify auto recomputation as well
    opts = popart.SessionOptions()
    opts.autoRecomputation = popart.RecomputationType.Standard

    with pytest.raises(popart.popart_exception) as e_info:
        session = popart.TrainingSession(
            fnModel=builder.getModelProto(),
            dataFeed=popart.DataFlow(1, {o: popart.AnchorReturnType("All")}),
            optimizer=popart.ConstSGD(0.001),
            losses=[popart.L1Loss(o, "l1LossVal", 0.1)],
            patterns=popart.Patterns([]),
            userOptions=opts,
            deviceInfo=tu.create_test_device())
    assert (e_info.value.args[0] ==
            "A mixture of auto and manual recomputaion is not supported")
