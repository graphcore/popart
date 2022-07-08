# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import popart
import pytest
import test_util as tu


# Both manual and auto recomputation are not supported
# concurrently
def test_valid_recompute_options():
    builder = popart.Builder()

    i1 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1]))
    r1 = builder.aiOnnx.relu([i1])
    o = builder.aiOnnx.relu([r1])

    # specify manual recomputation
    builder.recomputeOutputInBackwardPass(r1)

    # specify auto recomputation as well
    opts = popart.SessionOptions()
    opts.autoRecomputation = popart.RecomputationType.Standard

    with tu.create_test_device() as device:
        with pytest.raises(popart.popart_exception) as e_info:
            _ = popart.TrainingSession(
                fnModel=builder.getModelProto(),
                dataFlow=popart.DataFlow(1, [o]),
                optimizer=popart.ConstSGD(0.001),
                loss=o,
                patterns=popart.Patterns([]).enableRuntimeAsserts(False),
                userOptions=opts,
                deviceInfo=device,
            )
        assert (
            e_info.value.args[0]
            == "A mixture of auto and manual recomputaion is not supported"
        )
