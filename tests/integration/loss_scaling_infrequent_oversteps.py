# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
import onnx
import test_util as tu

from loss_scaling_util_test import getModelProto


def run_automatic_loss_scaling_infrequent_oversteps_test(
        tmpdir, update_period=1):
    """
    An integration test:
    We construct two sessions with an updatePeriod > 1.
    We run the first one with batchesPerStep == K * updatePeriod.
    We run the second one with batchesPerStep == 1, and we call
    session.run K * updatePeriod times.
    In the middle of this we checkpoint the model and load from the checkpoint
    to confirm the automatic loss scaling counter state is checkpointed.
    """
    loss, proto, t0, t_shape, label, label_shape, _ = getModelProto()

    opts = popart.SessionOptions()
    opts.automaticLossScalingSettings.enabled = True
    opts.automaticLossScalingSettings.binEdgeLocation = 0.5
    opts.automaticLossScalingSettings.thresholdUpperCountProportion = 0.2
    opts.automaticLossScalingSettings.updatePeriod = update_period

    init_optimizer = popart.SGD({
        "lossScaling": (10.0, False),
        "defaultMomentum": (0.5, False),
        "defaultVelocityScaling": (0.5, False),
        "defaultDampening": (0.5, False),
        "defaultWeightDecay": (0.5, False)
    })

    num_ipus = 1

    # The second session
    def getSession(device, modelPath=""):
        if modelPath == "":
            fnProto = proto
        else:
            fnProto = modelPath

        b_bps = 1
        b_session = popart.TrainingSession(fnModel=fnProto,
                                           deviceInfo=device,
                                           dataFlow=popart.DataFlow(b_bps, {}),
                                           loss=loss,
                                           optimizer=init_optimizer,
                                           userOptions=opts)
        b_session.prepareDevice()
        b_session.weightsFromHost()
        b_anchors = b_session.initAnchorArrays()
        return b_session, b_anchors

    # The first session
    with tu.create_test_device(num_ipus) as device:
        K = 1
        a_bps = K * update_period
        a_session = popart.TrainingSession(fnModel=proto,
                                           deviceInfo=device,
                                           dataFlow=popart.DataFlow(a_bps, {}),
                                           loss=loss,
                                           optimizer=init_optimizer,
                                           userOptions=opts)
        a_session.prepareDevice()
        a_session.weightsFromHost()
        a_anchors = a_session.initAnchorArrays()

        # Data
        a_data = np.random.rand(a_bps, *t_shape).astype(np.float16)
        a_label_data = np.random.randint(
            0, label_shape[0], a_bps * label_shape[0]).astype(np.int32)
        inputs = {t0: a_data, label: a_label_data}
        b_inputs = []
        for i in range(int(K * update_period)):
            b_inputs.append({
                t0:
                a_data[i],
                label:
                a_label_data[i * label_shape[0]:i * label_shape[0] +
                             label_shape[0]]
            })

        # Run and test.
        a_session.run(popart.PyStepIO(inputs, a_anchors))
        a_modelPath = str(tmpdir / "_a")
        a_session.modelToHost(a_modelPath)
    a_onnx = onnx.load(a_modelPath)
    a_counter_value = np.nan
    for _, weight in enumerate(a_onnx.graph.initializer):
        if "Histogram" in weight.name and "_counter" in weight.name:
            a_counter_value = weight.int32_data[0]

    b_modelPath = ""
    b_last_counter_value = np.nan
    for i in range(K * update_period):
        with tu.create_test_device(num_ipus) as device:
            b_session, b_anchors = getSession(device, modelPath=b_modelPath)
            b_session.run(popart.PyStepIO(b_inputs[i], b_anchors))
            model_file_name = str(i)
            b_modelPath = str(tmpdir / model_file_name)
            b_session.modelToHost(b_modelPath)
        b_onnx = onnx.load(b_modelPath)
        for _, weight in enumerate(b_onnx.graph.initializer):
            if "Histogram" in weight.name and "_counter" in weight.name:
                assert weight.int32_data[0] == i
                # Check that automatic loss scaling counter state is checkpointed
                # and persistent.
                b_last_counter_value = i

    # Check that automatic loss scaling counter states are same for the sessions.
    assert a_counter_value == b_last_counter_value
    assert a_counter_value == update_period - 1


def test_auto_loss_scaling_identical_weight_updates_update_period(tmpdir):
    run_automatic_loss_scaling_infrequent_oversteps_test(tmpdir,
                                                         update_period=4)
