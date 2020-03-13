# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import numpy as np
import pytest
import popart

# `import test_util` requires adding to sys.path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu

hidden_size = 16
batch_size = 32


@tu.requires_ipu_model
def test_pipeline_grad_accl_model1():
    """
    Test a sequence of matmuls with gradient accumulation and with/
    without pipelining.
    """
    np.random.seed(1234)
    gradAcclFactor = 8
    batchesPerStep = 8

    labelArray = np.random.randint(0, hidden_size, batch_size)
    gradaccl_no_pipeline_anchors = get_model_anchors_model1(
        doSharding=False,
        doPipelining=False,
        batchesPerStep=batchesPerStep,
        doTraining=True,
        doGradAccl=True,
        gradAcclFactor=gradAcclFactor,
        labelArray=labelArray)
    gradaccl_pipeline_anchors = get_model_anchors_model1(
        doSharding=True,
        doPipelining=True,
        batchesPerStep=batchesPerStep,
        doTraining=True,
        doGradAccl=True,
        gradAcclFactor=gradAcclFactor,
        labelArray=labelArray)
    for (tId1, t1), (tId2, t2) in zip(gradaccl_no_pipeline_anchors.items(),
                                      gradaccl_pipeline_anchors.items()):
        for i in range(min(np.shape(t1)[0], np.shape(t2)[0])):
            print("gradaccl no pipelining, batch: ", i, tId1, np.sum(t1[i]))
            print("gradaccl pipelining   , batch: ", i, tId2, np.sum(t2[i]))
        assert np.allclose(t1, t2)


@tu.requires_ipu_model
def test_pipeline_grad_accl_model2():
    """
    Test a non-linear model with gradient accumulation and with/
    without pipelining.
    """
    gradAcclFactor = 8
    batchesPerStep = 8

    gradaccl_no_pipeline_anchors = get_model_anchors_model2(
        doSharding=False,
        doPipelining=False,
        batchesPerStep=batchesPerStep,
        doTraining=True,
        doGradAccl=True,
        gradAcclFactor=gradAcclFactor)
    gradaccl_pipeline_anchors = get_model_anchors_model2(
        doSharding=True,
        doPipelining=True,
        batchesPerStep=batchesPerStep,
        doTraining=True,
        doGradAccl=True,
        gradAcclFactor=gradAcclFactor)
    for (tId1, t1), (tId2, t2) in zip(gradaccl_no_pipeline_anchors.items(),
                                      gradaccl_pipeline_anchors.items()):
        for i in range(min(np.shape(t1)[0], np.shape(t2)[0])):
            print("gradaccl no pipelining, batch: ", i, tId1, np.sum(t1[i]))
            print("gradaccl pipelining   , batch: ", i, tId2, np.sum(t2[i]))
        assert np.allclose(t1, t2)


@tu.requires_ipu_model
def test_invalid_grad_accl_size():
    """
    In this test we check that an error is thrown when using a gradient accumulation
    factor too small for the number of IPUs.
    """
    gradAcclFactor = 1
    batchesPerStep = 8

    with pytest.raises(popart.popart_exception) as e_info:
        get_model_anchors_model2(doSharding=True,
                                 doPipelining=True,
                                 batchesPerStep=batchesPerStep,
                                 doTraining=True,
                                 doGradAccl=True,
                                 gradAcclFactor=gradAcclFactor)
    assert e_info.value.args[0].startswith(
        "For pipelining, depth (gradient accumulation factor)")


def get_model_anchors_model1(doSharding,
                             doPipelining,
                             batchesPerStep,
                             doTraining,
                             doGradAccl=False,
                             gradAcclFactor=1,
                             doProfiling=False,
                             doDevicex=True,
                             anchorRestoredTensors=False,
                             labelArray=None):
    micro_batch_size = batch_size // gradAcclFactor
    builder = popart.Builder()

    input_shape = [micro_batch_size, hidden_size]
    input_ = builder.addInputTensor(popart.TensorInfo("FLOAT", input_shape))

    x = input_
    with builder.virtualGraph(0):
        for i in range(2):
            w = builder.addInitializedInputTensor(
                np.ones([hidden_size, hidden_size]).astype(np.float32),
                f"weight_0_{i}")
            x = builder.aiOnnx.matmul([x, w])
    with builder.virtualGraph(1 if doSharding else 0):
        for i in range(2):
            w = builder.addInitializedInputTensor(
                np.ones([hidden_size, hidden_size]).astype(np.float32),
                f"weight_1_{i}")
            x = builder.aiOnnx.matmul([x, w])
    with builder.virtualGraph(2 if doSharding else 0):
        for i in range(2):
            w = builder.addInitializedInputTensor(
                np.ones([hidden_size, hidden_size]).astype(np.float32),
                f"weight_2_{i}")
            if i == 1: w0 = w
            x = builder.aiOnnx.matmul([x, w])
    output = x

    builder.addOutputTensor(output)

    label_shape = [micro_batch_size]
    label = builder.addInputTensor(popart.TensorInfo("INT32", label_shape))

    art = popart.AnchorReturnType("ALL")
    losses = [popart.NllLoss(output, label, "NllLossVal")]

    # Loss on the last IPU
    losses[0].virtualGraph(2 if doSharding else 0)

    anchor_map = {losses[0].output(0): art, w0: art}
    if doTraining is True:
        anchor_map[popart.reservedGradientPrefix() + x] = art
        if doPipelining is True and anchorRestoredTensors is True:
            anchor_map[popart.reservedRestoredPrefix() + x] = art
            anchor_map[popart.reservedRestoredPrefix() + w0] = art
        if doGradAccl is True:
            anchor_map[popart.reservedAcclToUpdatePrefix() +
                       popart.reservedGradientPrefix() + w0] = art

    opts = popart.SessionOptions()
    opts.reportOptions = {"showExecutionSteps": "true"}
    opts.enablePipelining = doPipelining
    opts.enableGradientAccumulation = doGradAccl
    opts.accumulationFactor = gradAcclFactor
    opts.virtualGraphMode = popart.VirtualGraphMode.Manual

    if doSharding is False:
        numIPUs = 1
    else:
        numIPUs = 3

    if doTraining is True:
        session = popart.TrainingSession(
            fnModel=builder.getModelProto(),
            dataFeed=popart.DataFlow(batchesPerStep, anchor_map),
            losses=losses,
            optimizer=popart.ConstSGD(0.01),
            userOptions=opts,
            deviceInfo=tu.create_test_device(numIpus=numIPUs))
    else:
        session = popart.InferenceSession(
            fnModel=builder.getModelProto(),
            dataFeed=popart.DataFlow(batchesPerStep, anchor_map),
            losses=losses,
            userOptions=opts,
            deviceInfo=tu.create_test_device(numIpus=numIPUs))

    if doDevicex is False:
        return None

    anchors = session.initAnchorArrays()
    session.prepareDevice()

    outer_dim = 1
    if batchesPerStep > 1:
        # Add an outer dimension of batchesPerStep. We repeat the labels
        # as we want consistency if we have different shape inputs between examples.
        outer_dim *= batchesPerStep
        labelArray = np.repeat(labelArray[np.newaxis], batchesPerStep, 0)
    if gradAcclFactor > 1:
        # Divide up the batches per step batches into gradAcclFactor * batchesPerStep
        # samples.
        outer_dim *= gradAcclFactor
        labelArray = labelArray.reshape([gradAcclFactor * batchesPerStep, -1])
    if outer_dim > 1:
        # Add the gradAcclFactor * batchesPerStep dimension into the input.
        input_shape = [outer_dim] + input_shape

    stepio = popart.PyStepIO(
        {
            input_: np.ones(input_shape, np.float32),
            label: labelArray.astype(np.int32)
        }, anchors)

    session.weightsFromHost()
    if doTraining is True:
        session.optimizerFromHost()

    session.run(stepio)

    if doProfiling is True:
        from gcprofile import save_popart_report
        save_popart_report(session)

    return anchors


def get_model_anchors_model2(doSharding,
                             doPipelining,
                             batchesPerStep,
                             doTraining,
                             doGradAccl=False,
                             gradAcclFactor=1,
                             doProfiling=False,
                             doDevicex=True,
                             anchorRestoredTensors=False,
                             returnRawInput=False,
                             labelArray=None):

    np.random.seed(1234)
    builder = popart.Builder()
    micro_batch_size = batch_size // gradAcclFactor

    shape_d0 = [micro_batch_size, 2, 4, 4]
    shape_l0 = [batch_size]
    d0 = builder.addInputTensor(popart.TensorInfo("FLOAT", shape_d0), "inp")
    data_w0 = np.ones(shape=[2, 2, 3, 3]).astype(np.float32)
    w0 = builder.addInitializedInputTensor(data_w0, "weights")

    s0 = builder.aiOnnx.sin([d0], "s0")
    e0 = builder.aiOnnx.exp([s0], "e0")
    c0 = builder.aiOnnx.conv([e0, w0],
                             dilations=[1, 1],
                             pads=[1, 1, 1, 1],
                             strides=[1, 1],
                             debugPrefix="c0")
    r0 = builder.reshape_const(builder.aiOnnx, [c0], [micro_batch_size, 32])
    out = builder.aiOnnx.softmax([r0], axis=1, debugPrefix="sfm")
    builder.addOutputTensor(out)

    label_shape = [micro_batch_size]
    l0 = builder.addInputTensor(popart.TensorInfo("INT32", label_shape),
                                "label")

    art = popart.AnchorReturnType("ALL")
    loss = popart.NllLoss(out, l0, "loss")

    anchor_map = {"loss": art, w0: art, e0: art, s0: art, c0: art}
    if doTraining is True:
        anchor_map[popart.reservedGradientPrefix() + d0] = art
        if doPipelining is True and anchorRestoredTensors is True:
            anchor_map[popart.reservedRestoredPrefix() + e0] = art
            anchor_map[d0] = art
            anchor_map[popart.reservedRestoredPrefix() + d0] = art
        if doGradAccl is True:
            anchor_map[popart.reservedAcclToUpdatePrefix() +
                       popart.reservedGradientPrefix() + w0] = art

    opts = popart.SessionOptions()
    opts.reportOptions = {"showExecutionSteps": "true"}
    opts.enablePipelining = doPipelining
    opts.enableGradientAccumulation = doGradAccl
    opts.accumulationFactor = gradAcclFactor

    if doSharding is False:
        numIPUs = 1
    else:
        opts.virtualGraphMode = popart.VirtualGraphMode.Manual
        numIPUs = 3
        builder.virtualGraph(s0, 0)
        builder.virtualGraph(e0, 1)
        builder.virtualGraph(c0, 1)
        builder.virtualGraph(r0, 2)
        builder.virtualGraph(out, 2)
        loss.virtualGraph(2)

    if doTraining is True:
        session = popart.TrainingSession(
            fnModel=builder.getModelProto(),
            dataFeed=popart.DataFlow(batchesPerStep, anchor_map),
            losses=[loss],
            optimizer=popart.ConstSGD(0.01),
            userOptions=opts,
            deviceInfo=tu.create_test_device(numIpus=numIPUs))
    else:
        session = popart.InferenceSession(
            fnModel=builder.getModelProto(),
            dataFeed=popart.DataFlow(batchesPerStep, anchor_map),
            losses=[loss],
            userOptions=opts,
            deviceInfo=tu.create_test_device(numIpus=numIPUs))

    if doDevicex is False:
        return None

    anchors = session.initAnchorArrays()
    session.prepareDevice()

    classes = np.prod(shape_d0) / (micro_batch_size * batchesPerStep)
    label = np.random.randint(low=0, high=classes,
                              size=shape_l0).astype(np.int32)

    outer_dim = 1
    if batchesPerStep > 1:
        # Add an outer dimension of batchesPerStep. We repeat the labels
        # as we want consistency if we have different shape inputs between examples.
        outer_dim *= batchesPerStep
        label = np.repeat(label[np.newaxis], batchesPerStep, 0)
    if gradAcclFactor > 1:
        # Divide up the batches per step batches into gradAcclFactor * batchesPerStep
        # samples.
        outer_dim *= gradAcclFactor
        label = label.reshape([gradAcclFactor * batchesPerStep, -1])
    if outer_dim > 1:
        # Add the gradAcclFactor * batchesPerStep dimension into the input.
        shape_d0.insert(0, outer_dim)
    data = np.ones(shape=shape_d0).astype(np.float32)

    inputs = {d0: data, l0: label}
    stepio = popart.PyStepIO(inputs, anchors)

    session.weightsFromHost()
    if doTraining is True:
        session.optimizerFromHost()

    for i in range(6):
        session.run(stepio)

    if doProfiling is True:
        from gcprofile import save_popart_report
        save_popart_report(session)

    if returnRawInput is True:
        anchors["input_raw"] = data

    return anchors
