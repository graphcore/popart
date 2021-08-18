# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
import pytest


@pytest.mark.parametrize(
    "w_dtype,loss_dtype_str,loss_scaling",
    [
        (np.float16, 'FLOAT16', 1),  # should_clip
        (np.float32, 'FLOAT16', 1),  # should_clip
        (np.float16, 'FLOAT', 1),  # should_clip
        (np.float32, 'FLOAT', 1),
        (np.float16, 'FLOAT16', 2),  # should_clip
        (np.float32, 'FLOAT16', 2),  # should_clip
        (np.float16, 'FLOAT', 2),  # should_clip
        (np.float32, 'FLOAT', 2),
    ])
def test_auto_loss_scaling_clip_final_loss_scale(w_dtype, loss_dtype_str,
                                                 loss_scaling):
    """Test whether the final loss scale is correctly clipped at 2^15, when the 
    weights are in fp16 or the loss (and final loss scale) are in fp16. Also 
    check whether the up/down scaling are matched.

    To do this, a simple model is built as follows:
        W = [1] -- MatMul ------ MSE --- Loss
        X = [1] -'    Y = [1] -'

    Since the output of the model matches the labels, the loss, and consequently
    the gradients are all 0, which will cause the final loss scale to be doubled
    on every iteration. The test checks if the final loss scale is clipped if 
    the conditions above are met.
    """
    # Test params
    learning_rate = 0.1
    bps = 20  # Bathes per step
    should_clip = w_dtype == np.float16 or loss_dtype_str == 'FLOAT16'
    loss_dtype = np.float16 if loss_dtype_str == 'FLOAT16' else np.float32

    # Tensor names to anchor
    fls = popart.TensorId("finalLossScale")
    ils = popart.TensorId("finalInverseLossScale_0")
    lsf = popart.TensorId("lossScaleUpdateFactor")

    builder = popart.Builder()

    # Inputs, weights, labels.
    x = builder.addInitializedInputTensor(np.array([1], dtype=w_dtype))
    w = builder.addInitializedInputTensor(np.array([1], dtype=w_dtype))
    y = builder.addInitializedInputTensor(np.array([1], dtype=w_dtype))

    # Forward
    y_pred = builder.aiOnnx.matmul([x, w])

    # Loss
    e = builder.aiOnnx.sub([y, y_pred])  # error
    se = builder.aiOnnx.mul([e, e])  # squared error
    se = builder.aiOnnx.cast([se], loss_dtype_str)

    optimizer = popart.SGD({
        "defaultLearningRate": (learning_rate, False),
        "defaultMomentum": (0, True),
        "lossScaling": (loss_scaling, False)
    })

    opts = popart.SessionOptions()
    opts.automaticLossScalingSettings.enabled = True
    opts.engineOptions = {"debug.nanOverflowException": "false"}

    # Should run in Sim or Hw, because of fp16 not being accurate in IpuModel.
    device = popart.DeviceManager().createSimDevice({
        "numIPUs": 1,
        "tilesPerIPU": 4
    })

    session = popart.TrainingSession(
        fnModel=builder.getModelProto(),
        deviceInfo=device,
        dataFlow=popart.DataFlow(
            bps, {
                fls: popart.AnchorReturnType("All"),
                ils: popart.AnchorReturnType("All"),
                lsf: popart.AnchorReturnType("All"),
            }),
        loss=se,
        optimizer=optimizer,
        userOptions=opts,
    )

    session.prepareDevice()
    session.weightsFromHost()
    anchors = session.initAnchorArrays()
    session.run(popart.PyStepIO({}, anchors))

    expected_lsf = np.geomspace(2**0, 2**(bps - 1), num=bps, dtype=loss_dtype)
    if should_clip:
        clip_at = np.array(2**15 / loss_scaling, dtype=loss_dtype)
        expected_lsf = np.clip(expected_lsf, 0, clip_at)
    expected_fls = expected_lsf * loss_scaling
    expected_ils = (learning_rate / loss_scaling) / expected_lsf.astype(
        np.float32)

    assert np.allclose(anchors[lsf], expected_lsf)
    assert np.allclose(anchors[fls], expected_fls)
    assert np.allclose(anchors[ils], expected_ils)
