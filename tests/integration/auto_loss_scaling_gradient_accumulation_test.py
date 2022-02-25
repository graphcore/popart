# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
import pytest
import test_util as tu

from loss_scaling_util_test import getModelProto


@pytest.mark.parametrize("gradientTensorTrackingMethod", [
    popart.GradientTensorTrackingMethod.ConvAndMatmulGradients,
    popart.GradientTensorTrackingMethod.AllNonViewChangingGradientTensors
])
def test_loss_scale_updates_with_grad_accumulation_correctness(
        gradientTensorTrackingMethod):
    """
    Run a session that has gradient accumulation and auto loss scaling
    enabled. Verify that:
    1. the loss scale is updated in the outer loop
    2. the gradient statistics are accumulated in the inner loop
    """
    init_ls = 1000.0
    optimizer = popart.SGD({"lossScaling": (init_ls, False)})
    accumulation_factor = 5

    opts = popart.SessionOptions()
    opts.automaticLossScalingSettings.enabled = True
    opts.automaticLossScalingSettings.binEdgeLocation = 0.5
    opts.automaticLossScalingSettings.thresholdUpperCountProportion = 0.2
    opts.automaticLossScalingSettings.gradientTensorTrackingMethod = gradientTensorTrackingMethod
    opts.enableGradientAccumulation = True
    opts.accumulationFactor = accumulation_factor

    loss, proto, t0, t_shape, label, label_shape, _ = getModelProto()
    bps = 4

    accl_stats_id = "Accum___autoLossScaleStats"
    ls_id = "finalLossScale"
    session = popart.TrainingSession(fnModel=proto,
                                     deviceInfo=tu.create_test_device(),
                                     dataFlow=popart.DataFlow(
                                         bps, [accl_stats_id, ls_id]),
                                     loss=loss,
                                     optimizer=optimizer,
                                     userOptions=opts)
    session.prepareDevice()
    session.weightsFromHost()
    anchors = session.initAnchorArrays()

    step_size = bps * accumulation_factor
    t0_data = 10000.0 * np.random.rand(step_size, *t_shape).astype(np.float16)
    label_data = np.random.randint(0, label_shape[0],
                                   step_size * label_shape[0]).astype(np.int32)
    inputs = {t0: t0_data, label: label_data}

    session.run(popart.PyStepIO(inputs, anchors))

    # 1.
    prev_ls0 = init_ls
    for i in range(len(anchors[ls_id])):
        ls0 = anchors[ls_id][i][0]
        for j in range(len(anchors[ls_id][0])):
            # loss scale is the same within inner loops
            assert anchors[ls_id][i][j] == ls0

        if i > 0:
            # loss scale has been updated between outer loops
            assert ls0 != prev_ls0

    # 2.
    for batch_stats in anchors[accl_stats_id]:
        for minibatch_idx, minibatch in enumerate(batch_stats):
            # how many elements are there in the grad tensors to be tracked?
            # np.prod(t_shape) for each of:
            #  - conv weights grad
            #  - conv data grad
            #  - matmul weights grad
            #  - relu grad
            # And if 'gradientTensorTrackingMethod' is
            # 'AllNonViewChangingGradientTensors', then also
            #  - relu grad
            if gradientTensorTrackingMethod == popart.GradientTensorTrackingMethod.ConvAndMatmulGradients:
                num_tracked_tensors = 3
            elif gradientTensorTrackingMethod == popart.GradientTensorTrackingMethod.AllNonViewChangingGradientTensors:
                num_tracked_tensors = 4

            num_elms_gradstats = num_tracked_tensors * np.prod(t_shape)
            assert np.sum(minibatch) == num_elms_gradstats * (minibatch_idx +
                                                              1)
