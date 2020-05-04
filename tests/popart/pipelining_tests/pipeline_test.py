# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
import pytest
import re

# importing test_session and test_util requires adding to sys.path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from test_session import PopartTestSession
import test_util as tu


@tu.requires_ipu_model
def test_disabled_virtual_graphs():
    """
    In this test we check that an error is thrown when doing pipelining
    if enableVirtualGraph session option is not set to true
    """
    builder, op0_out, op1_out, op2_out, op3_out, anchor_map, loss = get_simple_linear_model(
    )

    opts = popart.SessionOptions()
    opts.enablePipelining = True
    opts.virtualGraphMode = popart.VirtualGraphMode.Off

    with pytest.raises(popart.popart_exception) as e_info:
        session = popart.InferenceSession(fnModel=builder.getModelProto(),
                                          dataFeed=popart.DataFlow(
                                              10, anchor_map),
                                          userOptions=opts,
                                          losses=[loss],
                                          deviceInfo=tu.create_test_device())
    assert e_info.value.args[0].startswith("Pipelining requires more than")


@tu.requires_ipu_model
def test_one_ipu():
    """
    In this test we check that an error is thrown when doing pipelining
    on 1 IPU
    """
    builder = popart.Builder()
    shape_d = [10]
    shape_l = [1]
    d0 = builder.addInputTensor(popart.TensorInfo("FLOAT", shape_d))
    d1 = builder.addInputTensor(popart.TensorInfo("FLOAT", shape_d))
    l0 = builder.addInputTensor(popart.TensorInfo("INT32", shape_l))
    op0_out = builder.aiOnnx.sin([d0], "s0")
    op1_out = builder.aiOnnx.exp([d1], "r0")
    op2_out = builder.aiOnnx.mul([op0_out, op1_out], "m0")
    builder.addOutputTensor(op2_out)
    loss = popart.NllLoss(op2_out, l0, "loss")
    opts = popart.SessionOptions()
    opts.enablePipelining = True
    opts.virtualGraphMode = popart.VirtualGraphMode.Manual  # i.e. use 1 ipu
    builder.pipelineStage(op0_out, 0)
    builder.virtualGraph(op0_out, 0)
    builder.pipelineStage(op1_out, 0)
    builder.virtualGraph(op1_out, 0)
    builder.pipelineStage(op2_out, 1)
    builder.virtualGraph(op2_out, 0)
    loss.pipelineStage(1)
    loss.virtualGraph(0)
    with pytest.raises(popart.popart_exception) as e_info:
        session = popart.InferenceSession(fnModel=builder.getModelProto(),
                                          dataFeed=popart.DataFlow(
                                              10, [op2_out, "loss"]),
                                          userOptions=opts,
                                          losses=[loss],
                                          deviceInfo=tu.create_test_device())
        session.prepareDevice()
    assert e_info.value.args[0].startswith("Pipelining requires more than")


@tu.requires_ipu_model
def test_enabled_recomputation():
    """
    In this test we check that NO error is thrown when doing pipelining
    if recomputation is enabled
    """
    builder, op0_out, op1_out, op2_out, op3_out, anchor_map, loss = get_simple_linear_model(
    )

    opts = popart.SessionOptions()
    opts.enablePipelining = True
    opts.virtualGraphMode = popart.VirtualGraphMode.Manual
    opts.autoRecomputation = popart.RecomputationType.Standard

    builder.virtualGraph(op0_out, 0)
    builder.virtualGraph(op1_out, 1)
    builder.virtualGraph(op2_out, 1)
    builder.virtualGraph(op3_out, 1)
    loss.virtualGraph(1)

    session = popart.InferenceSession(fnModel=builder.getModelProto(),
                                      dataFeed=popart.DataFlow(10, anchor_map),
                                      userOptions=opts,
                                      losses=[loss],
                                      deviceInfo=tu.create_test_device(
                                          numIpus=2, tilesPerIpu=20))


def test_stream_tensors_to_multiple_ipus():
    """
    Streaming an input to Ops on multiple IPUs throws an error

    09/07/2019 Since D12445 this test no longer raises an exception. By
    default, stream tensors are now replicated by streaming to a single
    IPU, then copied across to the other IPUs where they are needed.
    Leaving this test in to verify that this remains the case
    """
    builder, op0_out, op1_out, op2_out, op3_out, anchor_map, loss = get_simple_linear_model(
        streamInputToOp1AndOp2=True)

    opts = popart.SessionOptions()
    opts.enablePipelining = True
    opts.virtualGraphMode = popart.VirtualGraphMode.Manual

    builder.virtualGraph(op0_out, 0)
    builder.virtualGraph(op1_out, 1)
    builder.virtualGraph(op2_out, 1)
    builder.virtualGraph(op3_out, 1)
    loss.virtualGraph(1)

    session = popart.InferenceSession(fnModel=builder.getModelProto(),
                                      dataFeed=popart.DataFlow(10, anchor_map),
                                      userOptions=opts,
                                      losses=[loss],
                                      deviceInfo=tu.create_test_device(
                                          numIpus=2, tilesPerIpu=20))


def test_sharding_multi_source():
    """
    Branched sharding does not merge IPU Copies with pipelining
    e.g. Op0 -> Op2
                 ^
         Op1 ----'
    where the vGraph split is IPU0 : {Op0}, IPU1 : {Op1}, IPU2 : {Op2}
    """
    builder = popart.Builder()
    shape_d = [10]
    shape_l = [1]
    d0 = builder.addInputTensor(popart.TensorInfo("FLOAT", shape_d))
    d1 = builder.addInputTensor(popart.TensorInfo("FLOAT", shape_d))
    l0 = builder.addInputTensor(popart.TensorInfo("INT32", shape_l))

    op0_out = builder.aiOnnx.sin([d0], "s0")
    op1_out = builder.aiOnnx.exp([d1], "r0")
    op2_out = builder.aiOnnx.mul([op0_out, op1_out], "m0")
    builder.addOutputTensor(op2_out)

    art = popart.AnchorReturnType("All")
    loss = popart.NllLoss(op2_out, l0, "loss")
    anchor_map = {op2_out: art, "loss": art}

    opts = popart.SessionOptions()
    opts.enablePipelining = True
    opts.virtualGraphMode = popart.VirtualGraphMode.Manual

    builder.virtualGraph(op0_out, 0)
    builder.virtualGraph(op1_out, 1)
    builder.virtualGraph(op2_out, 2)
    loss.virtualGraph(2)

    session = popart.InferenceSession(fnModel=builder.getModelProto(),
                                      dataFeed=popart.DataFlow(10, anchor_map),
                                      userOptions=opts,
                                      losses=[loss],
                                      deviceInfo=tu.create_test_device(
                                          numIpus=3, tilesPerIpu=20))


def test_inference_min_batches():
    """
    Check that we throw if too few batches to fill and flush the pipeline
    for an inference model
    """
    minBatches = 3  # == numIpus == numPipelineStages

    get_model_anchors(doSharding=True,
                      doPipelining=True,
                      batchesPerStep=minBatches,
                      doTraining=False,
                      doDevicex=False)

    with pytest.raises(popart.popart_exception) as e_info:
        get_model_anchors(doSharding=True,
                          doPipelining=True,
                          batchesPerStep=minBatches - 1,
                          doTraining=False,
                          doDevicex=False)
    assert e_info.value.args[0].startswith(
        "For pipelining, depth (batchesPerStep) must")


def test_training_min_batches():
    """
    Check that we throw if too few batches to fill and flush the pipeline
    for a training model
    """
    minBatches = 5  # == 2 * (numIpus-1) + 1 == numPipelineStages

    get_model_anchors(doSharding=True,
                      doPipelining=True,
                      batchesPerStep=minBatches,
                      doTraining=True,
                      doDevicex=False)

    with pytest.raises(popart.popart_exception) as e_info:
        get_model_anchors(doSharding=True,
                          doPipelining=True,
                          batchesPerStep=minBatches - 1,
                          doTraining=True,
                          doDevicex=False)
    assert e_info.value.args[0].startswith(
        "For pipelining, depth (batchesPerStep) must")


def test_output_matches_train():
    """
    In this test we check that the anchors of equivalent non-sharded, sharded
    and non-pipelined, and sharded and pipelined models are equal when doing
    training. We expect only the first output and weight update to be the same
    as non-pipelined models
    """
    bps = 8
    singleIpu_anchors = get_model_anchors(doSharding=False,
                                          doPipelining=False,
                                          batchesPerStep=bps,
                                          doTraining=True)
    multiIpu_anchors = get_model_anchors(doSharding=True,
                                         doPipelining=False,
                                         batchesPerStep=bps,
                                         doTraining=True)
    pipelined_anchors = get_model_anchors(doSharding=True,
                                          doPipelining=True,
                                          batchesPerStep=bps,
                                          doTraining=True)
    # TODO, depends on T9630, add a case with grad accumulation. All tensor
    # outputs should be exactly the same when doing pipelined vs non-pipelined
    # when grad accumulation is turned on

    for (tId1, t1), (tId2, t2) in zip(singleIpu_anchors.items(),
                                      multiIpu_anchors.items()):
        assert np.allclose(t1, t2)

    # Expect only the anchors from the first batch to be equal. After that, the
    # continuous gradient accumulation option causes model parameters to diverge
    for (tId1, t1), (tId2, t2) in zip(singleIpu_anchors.items(),
                                      pipelined_anchors.items()):
        for i in range(np.shape(t1)[0]):
            print("singleIpu   , batch: ", i, tId1, np.sum(t1[i]))
            print("pipelinedIpu, batch: ", i, tId2, np.sum(t2[i]))
        assert np.allclose(t1[0], t2[0])


def test_acts_match_restored_acts():
    """
    In this test we check that the stashed tensors and their equivalent
    Restored tensors have the same values for all batches. This confirms
    that the schedule of restoring and streaming anchors is correct

    How do we know they're not both wrong? Take this example where the
    streamed input is stashed. Check that it matches the raw data input
    that is fed to the StepIO
    """
    bps = 8
    pipelined_anchors = get_model_anchors(doSharding=True,
                                          doPipelining=True,
                                          batchesPerStep=bps,
                                          doTraining=True,
                                          anchorRestoredTensors=True,
                                          returnRawInput=True)

    for (tId, t) in pipelined_anchors.items():
        for i in range(np.shape(t)[0]):
            print("batch: ", i, tId, np.sum(t[i]))

    assert np.allclose(
        pipelined_anchors[popart.reservedRestoredPrefix() + "Exp:0"],
        pipelined_anchors["Exp:0"])
    assert np.allclose(
        pipelined_anchors[popart.reservedRestoredPrefix() + "input"],
        pipelined_anchors["input"])
    assert np.allclose(pipelined_anchors["input_raw"],
                       pipelined_anchors["input"])


def test_output_matches_infer():
    """
    In this test we check that the anchors of equivalent non-sharded, sharded
    and non-pipelined, and sharded and pipelined models are equal when doing
    inference
    """
    bps = 8
    singleIpu_anchors = get_model_anchors(doSharding=False,
                                          doPipelining=False,
                                          batchesPerStep=bps,
                                          doTraining=False)
    multiIpu_anchors = get_model_anchors(doSharding=True,
                                         doPipelining=False,
                                         batchesPerStep=bps,
                                         doTraining=False)
    pipelined_anchors = get_model_anchors(doSharding=True,
                                          doPipelining=True,
                                          batchesPerStep=bps,
                                          doTraining=False)

    for (tId1, t1), (tId2, t2) in zip(singleIpu_anchors.items(),
                                      multiIpu_anchors.items()):
        for i in range(np.shape(t1)[0]):
            print("singleIpu, batch: ", i, tId1, np.sum(t1[i]))
            print("multiIpu , batch: ", i, tId2, np.sum(t2[i]))
        assert np.allclose(t1, t2)

    for (tId1, t1), (tId2, t2) in zip(singleIpu_anchors.items(),
                                      pipelined_anchors.items()):
        for i in range(np.shape(t1)[0]):
            print("singleIpu   , batch: ", i, tId1, np.sum(t1[i]))
            print("pipelinedIpu, batch: ", i, tId2, np.sum(t2[i]))
        assert np.allclose(t1, t2)


## TODO T8803 : requires hardware or a sim device
def test_pipelined_dropout():
    # The test can be run without pipelining for debugging.
    def test(do_pipelining, do_sharding):
        dsize = 10
        ratio = 0.5
        if do_sharding:
            ipus = 4
        else:
            ipus = 1
        layers = 4
        batches_per_step = 7

        # Ensure inputs in range [1.0, 2.0] to ensure comparing with 0 is valid
        ip_shape = [dsize]
        ip_data = np.full([batches_per_step] + ip_shape, 1).astype(np.float32)

        dropouts = []
        dropoutGrads = []
        dropoutInputs = []
        dropoutOutputs = []

        builder = popart.Builder()
        ip = builder.addInputTensor(popart.TensorInfo("FLOAT", ip_shape))

        def add_layer(layer_input, vgraph_num):
            # This is to get the output of the dropout in the bwd pass.
            # D_next_layer_in also includes the gradient of the AddOp.
            identity0 = builder.aiOnnx.identity([layer_input])
            if do_sharding:
                builder.virtualGraph(identity0, vgraph_num)

            [dropout0] = builder.aiOnnx.dropout([identity0],
                                                num_outputs=1,
                                                ratio=ratio)
            if do_sharding:
                builder.virtualGraph(dropout0, vgraph_num)

            # the input to the forward pass dropout
            dropoutInputs.append(identity0)
            # the input to the backward pass dropout
            dropoutInputs.append(popart.reservedGradientPrefix() + dropout0)
            # the output of the backward pass dropout
            dropoutGrads.append(popart.reservedGradientPrefix() + identity0)
            # the output of the forward pass dropout
            dropouts.append(dropout0)

            # This ensures the all input elements to the dropouts, in both
            # the forward and backward passes, will be non-zero.
            add0 = builder.aiOnnx.add([layer_input, dropout0])
            if do_sharding:
                builder.virtualGraph(add0, vgraph_num)

            return add0

        # construct a graph of `layers` number of layers
        # with each layer on a different IPU.
        next_layer_in = ip
        for vgraph in range(layers):
            next_layer_in = add_layer(next_layer_in, vgraph)
        out = next_layer_in
        builder.addOutputTensor(out)

        # TODO: use the tu.requires_ipu decorator
        if tu.ipu_available(ipus):
            device = tu.create_test_device(numIpus=ipus)
        else:
            pytest.skip("Test needs to run on IPU, but none are available")

        dfAnchors = {}
        for x in dropouts + dropoutGrads + dropoutInputs:
            dfAnchors[x] = popart.AnchorReturnType("All")

        dataFlow = popart.DataFlow(batches_per_step, dfAnchors)

        loss = popart.L1Loss(out, "l1LossVal", 0.1)
        if do_sharding:
            loss.virtualGraph(layers - 1)

        userOptions = popart.SessionOptions()
        if do_sharding:
            userOptions.virtualGraphMode = popart.VirtualGraphMode.Manual
        userOptions.enablePipelining = do_pipelining

        session = popart.TrainingSession(fnModel=builder.getModelProto(),
                                         dataFeed=dataFlow,
                                         optimizer=popart.ConstSGD(0.1),
                                         losses=[loss],
                                         userOptions=userOptions,
                                         deviceInfo=device)

        session.prepareDevice()
        session.weightsFromHost()
        anchors = session.initAnchorArrays()
        session.setRandomSeed(0)

        stepio = popart.PyStepIO({ip: ip_data}, anchors)

        session.run(stepio)

        print(anchors.keys())

        # Check that none of the elements of the dropout inputs are zero
        for tid in dropoutInputs:
            x = anchors[tid]
            print(f'{tid}: {x}')
            zero = np.zeros(x.shape)
            assert not np.any(np.equal(x, zero)), \
                   f'Some elements of dropout input {tid} are zero'

        print()

        # For each dropout, check that the masked out elements are the same
        # in the forward and backward passes.
        for fwdId, bwdId in zip(dropouts, dropoutGrads):
            print(f'{fwdId}:\n{np.sign(anchors[fwdId])}')
            print(f'{bwdId}:\n{np.sign(anchors[bwdId])}')
            lhs = np.sign(anchors[fwdId])
            rhs = np.sign(anchors[bwdId])
            assert np.array_equal(lhs, rhs), \
                   f'{fwdId} and {bwdId} did not use the same dropout mask'
            print()

        return anchors

    # Compare pipelined and non-pipelined models to confirm random
    # dropout behaviour is the same
    # Note: Because the random behaviour of ops such as dropout depend on a
    # reference tensor layout, which is not constant between these models
    # (since they all have a different IR), this behaviour is not always
    # guaranteed. However, we have chosen a model where the random behaviour
    # is the same between non-sharded, sharded, and sharded-pipelined
    # configurations
    singleipu_anchors = test(do_sharding=False, do_pipelining=False)
    shard_anchors = test(do_sharding=True, do_pipelining=False)
    pipe_anchors = test(do_sharding=True, do_pipelining=True)
    for tId in singleipu_anchors:
        assert np.array_equal(singleipu_anchors[tId], shard_anchors[tId])
        assert np.array_equal(singleipu_anchors[tId], pipe_anchors[tId])


## TODO T8803 : requires hardware or a sim device
def test_pipelined_recomputed_dropout():
    dsize = 10
    ratio = 0.5
    ipus = 4
    layers = 4
    batches_per_step = 7

    # Ensure inputs in range [1.0, 2.0] to ensure comparing with 0 is valid
    ip_shape = [dsize]
    ip_data = np.full([batches_per_step] + ip_shape, 1).astype(np.float32)

    dropouts = []
    dropoutGrads = []
    dropoutInputs = []
    dropoutOutputs = []

    builder = popart.Builder()
    ip = builder.addInputTensor(popart.TensorInfo("FLOAT", ip_shape))

    def add_layer(layer_input, vgraph_num):
        # This is to get the output of the dropout in the bwd pass.
        # D_next_layer_in also includes the gradient of the AddOp.
        identity0 = builder.aiOnnx.identity([layer_input])
        builder.virtualGraph(identity0, vgraph_num)

        [dropout0] = builder.aiOnnx.dropout([identity0],
                                            num_outputs=1,
                                            ratio=ratio)
        builder.virtualGraph(dropout0, vgraph_num)

        # the input to the forward pass dropout
        dropoutInputs.append(identity0)
        # the input to the backward pass dropout
        dropoutInputs.append(popart.reservedGradientPrefix() + dropout0)
        # the output of the backward pass dropout
        dropoutGrads.append(popart.reservedGradientPrefix() + identity0)
        # the output of the forward pass dropout
        dropouts.append(dropout0)

        relu0 = builder.aiOnnx.relu([dropout0])
        builder.virtualGraph(relu0, vgraph_num)

        # This ensures the all input elements to the dropouts, in both
        # the forward and backward passes, will be non-zero.
        add0 = builder.aiOnnx.add([layer_input, dropout0])
        builder.virtualGraph(add0, vgraph_num)

        return add0

    # construct a graph of `layers` number of layers
    # with each layer on a different IPU.
    next_layer_in = ip
    for vgraph in range(layers):
        next_layer_in = add_layer(next_layer_in, vgraph)
    out = next_layer_in
    builder.addOutputTensor(out)

    # TODO: use the tu.requires_ipu decorator
    if tu.ipu_available(ipus):
        device = tu.create_test_device(numIpus=ipus)
    else:
        pytest.skip("Test needs to run on IPU, but none are available")

    dfAnchors = {}
    for x in dropouts + dropoutGrads + dropoutInputs:
        dfAnchors[x] = popart.AnchorReturnType("All")

    dataFlow = popart.DataFlow(batches_per_step, dfAnchors)

    loss = popart.L1Loss(out, "l1LossVal", 0.1)
    loss.virtualGraph(layers - 1)

    userOptions = popart.SessionOptions()
    userOptions.virtualGraphMode = popart.VirtualGraphMode.Manual
    userOptions.enablePipelining = True
    userOptions.autoRecomputation = popart.RecomputationType.Pipeline

    session = popart.TrainingSession(fnModel=builder.getModelProto(),
                                     dataFeed=dataFlow,
                                     optimizer=popart.ConstSGD(0.1),
                                     losses=[loss],
                                     userOptions=userOptions,
                                     deviceInfo=device)

    session.prepareDevice()
    session.weightsFromHost()
    anchors = session.initAnchorArrays()
    session.setRandomSeed(0)

    stepio = popart.PyStepIO({ip: ip_data}, anchors)

    session.run(stepio)

    print(anchors.keys())

    # Check that none of the elements of the dropout inputs are zero
    for tid in dropoutInputs:
        x = anchors[tid]
        print(f'{tid}: {x}')
        zero = np.zeros(x.shape)
        assert not np.any(np.equal(x, zero)), \
               f'Some elements of dropout input {tid} are zero'

    print()

    # For each dropout, check that the masked out elements are the same
    # in the forward and backward passes.
    for fwdId, bwdId in zip(dropouts, dropoutGrads):
        print(f'{fwdId}:\n{np.sign(anchors[fwdId])}')
        print(f'{bwdId}:\n{np.sign(anchors[bwdId])}')
        lhs = np.sign(anchors[fwdId])
        rhs = np.sign(anchors[bwdId])
        assert np.array_equal(lhs, rhs), \
               f'{fwdId} and {bwdId} did not use the same dropout mask'
        print()


# Model
#  <--- ipu0 ----> <--------- ipu1 ---> <------------ ipu2 ------------>
#
#  d0 --|-- Sin --|-- Exp --|
#                           |-- Conv --|-- Reshape --|-- Softmax --> out
#                      w0 --|
def get_model_anchors(doSharding,
                      doPipelining,
                      batchesPerStep,
                      doTraining,
                      doProfiling=False,
                      doDevicex=True,
                      anchorRestoredTensors=False,
                      returnRawInput=False):
    np.random.seed(seed=1)

    builder = popart.Builder()
    batchSize = 2
    shape_d0 = [batchSize, 2, 4, 4]
    shape_l0 = [batchSize]
    d0 = builder.addInputTensor(popart.TensorInfo("FLOAT", shape_d0))
    data_w0 = np.ones(shape=[2, 2, 3, 3]).astype(np.float32)
    w0 = builder.addInitializedInputTensor(data_w0)
    l0 = builder.addInputTensor(popart.TensorInfo("INT32", shape_l0))

    s0 = builder.aiOnnx.sin([d0], "s0")
    e0 = builder.aiOnnx.exp([s0], "e0")
    c0 = builder.aiOnnx.conv([e0, w0],
                             dilations=[1, 1],
                             pads=[1, 1, 1, 1],
                             strides=[1, 1],
                             debugPrefix="c0")
    r0 = builder.reshape_const(builder.aiOnnx, [c0], [batchSize, 32])
    out = builder.aiOnnx.softmax([r0], axis=1, debugPrefix="sfm")
    builder.addOutputTensor(out)

    art = popart.AnchorReturnType("All")
    loss = popart.NllLoss(out, l0, "loss")

    anchor_map = {"loss": art, w0: art, e0: art}
    if doTraining is True:
        anchor_map[popart.reservedGradientPrefix() + d0] = art
        if doPipelining is True and anchorRestoredTensors is True:
            anchor_map[popart.reservedRestoredPrefix() + e0] = art
            anchor_map[d0] = art
            anchor_map[popart.reservedRestoredPrefix() + d0] = art

    opts = popart.SessionOptions()
    opts.reportOptions = {"showExecutionSteps": "true"}
    opts.enablePipelining = doPipelining

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
            deviceInfo=tu.create_test_device(numIpus=numIPUs, tilesPerIpu=20))
    else:
        session = popart.InferenceSession(
            fnModel=builder.getModelProto(),
            dataFeed=popart.DataFlow(batchesPerStep, anchor_map),
            losses=[loss],
            userOptions=opts,
            deviceInfo=tu.create_test_device(numIpus=numIPUs, tilesPerIpu=20))

    if doDevicex is False:
        return None

    anchors = session.initAnchorArrays()
    session.prepareDevice()

    if batchesPerStep > 1:
        shape_d0.insert(0, batchesPerStep)
        shape_l0.insert(0, batchesPerStep)
    data = np.random.uniform(low=-10.0, high=10.0,
                             size=shape_d0).astype(np.float32)
    classes = np.prod(shape_d0) / (batchSize * batchesPerStep)
    label = np.random.randint(low=0, high=classes,
                              size=shape_l0).astype(np.int32)

    inputs = {d0: data, l0: label}
    stepio = popart.PyStepIO(inputs, anchors)

    session.weightsFromHost()
    if doTraining is True:
        session.optimizerFromHost()
    session.run(stepio)

    if doProfiling is True:
        from gcprofile import save_popart_report
        save_popart_report(session)

    if returnRawInput is True:
        anchors["input_raw"] = data

    return anchors


def get_simple_linear_model(streamInputToOp1AndOp2=False):
    builder = popart.Builder()
    shape_d = [10]
    shape_l = [1]
    d0 = builder.addInputTensor(popart.TensorInfo("FLOAT", shape_d))
    d1 = builder.addInputTensor(popart.TensorInfo("FLOAT", shape_d))
    l0 = builder.addInputTensor(popart.TensorInfo("INT32", shape_l))

    op0_out = builder.aiOnnx.sin([d0], "s0")
    if streamInputToOp1AndOp2 is True:
        op1_out = builder.aiOnnx.mul([op0_out, d0])
    else:
        op1_out = builder.aiOnnx.mul([op0_out, d1])
    op2_out = builder.aiOnnx.exp([op1_out], "e0")
    op3_out = builder.aiOnnx.exp([op2_out], "e1")
    builder.addOutputTensor(op3_out)

    art = popart.AnchorReturnType("All")
    loss = popart.NllLoss(op3_out, l0, "loss")
    anchor_map = {op3_out: art, "loss": art}

    return builder, op0_out, op1_out, op2_out, op3_out, anchor_map, loss


def test_pipeline_stage_errors():
    dummy_data = np.zeros(2, dtype=np.float32)
    bps = 2

    vgraph_ids = []
    ps_ids = []

    def init_builder(builder):
        d0 = builder.addInputTensor(dummy_data, 'data0')
        d1 = builder.addInputTensor(dummy_data, 'data1')
        d2 = builder.addInputTensor(dummy_data, 'data2')

        s0 = builder.aiOnnx.sin([d0], "s0")
        m0 = builder.aiOnnx.mul([s0, d0])
        e0 = builder.aiOnnx.exp([m0])
        e1 = builder.aiOnnx.exp([e0])

        builder.addOutputTensor(e1)

        print(f'Setting virtual graphs to {vgraph_ids}')
        for tid, vgid in zip((s0, m0, e0, e1), vgraph_ids):
            if vgid is not None:
                builder.virtualGraph(tid, vgid)

        print(f'Setting pipeline stages to {ps_ids}')
        for tid, psid in zip((s0, m0, e0, e1), ps_ids):
            if psid is not None:
                builder.pipelineStage(tid, psid)

        loss = builder.addL1Loss(e1, 'l1LossVal', 0.1,
                                 popart.ReductionType.Sum)
        loss.virtualGraph(1)

        return [e1]

    session = PopartTestSession()
    session.options.enableVirtualGraphs = True
    session.options.enablePipelining = True
    session.device = 'ipu_model'
    session.numIPUs = 2
    session.batchesPerStep = bps

    # test a pipeline stage appearing on multiple virtual graphs
    vgraph_ids = [0, 0, 0, 1]
    ps_ids = [0, 0, 1, 1]
    with pytest.raises(popart.popart_exception) as e_info:
        session.prepare(init_builder)

    emsg = e_info.value.args[0]
    assert re.match('Ops .* have the same pipeline stage 1,.*',
                    emsg) is not None

    # test not all ops having a pipeline stage set
    vgraph_ids = [0, 0, 1, 1]
    ps_ids = [0, 0, None, 1]
    with pytest.raises(popart.popart_exception) as e_info:
        session.prepare(init_builder)

    emsg = e_info.value.args[0]
    assert emsg.startswith('Only some ops have had their pipeline stage set.')


def test_pipeline_stages_backwards_through_ipus():
    dummy_data = np.array([0.5, 1.0], dtype=np.float32)
    bps = 2

    vgraph_ids = []
    ps_ids = []

    def init_builder(builder):
        d0 = builder.addInputTensor(dummy_data, 'data0')

        s0 = builder.aiOnnx.sin([d0], "s0")
        m0 = builder.aiOnnx.mul([s0, d0])
        e0 = builder.aiOnnx.exp([m0])
        e1 = builder.aiOnnx.exp([e0], 'output')

        builder.addOutputTensor(e1)

        stage0 = [s0, m0]
        stage1 = [e0, e1]

        stage0_vgraph = 1
        stage1_vgraph = 0

        for tid in stage0:
            builder.virtualGraph(tid, stage0_vgraph)
            builder.pipelineStage(tid, 0)

        for tid in stage1:
            builder.virtualGraph(tid, stage1_vgraph)
            builder.pipelineStage(tid, 1)

        loss = builder.addL1Loss(e1, 'l1LossVal', 0.1,
                                 popart.ReductionType.Sum)
        loss.virtualGraph(stage1_vgraph)
        loss.pipelineStage(1)

        return [e1]

    def ref():
        d0 = dummy_data
        s0 = np.sin(d0)
        m0 = s0 * d0
        e0 = np.exp(m0)
        e1 = np.exp(e0)
        return e1

    session = PopartTestSession()
    session.options.enableVirtualGraphs = True
    session.options.enablePipelining = True
    session.device = 'ipu_model'
    session.numIPUs = 2
    session.batchesPerStep = bps

    # test a pipeline stage appearing on multiple virtual graphs
    session.prepare(init_builder)
    pipelineAnchors = session.run()

    assert len(pipelineAnchors) == 1
    pipelineAnchors = [v for k, v in pipelineAnchors.items()]
    pipelineAnchor = pipelineAnchors[0]

    print(pipelineAnchor)
    print(ref())

    assert np.allclose(pipelineAnchor[0], ref())


def test_multiple_stages_per_virtual_graph_inference():
    bps = 4
    dummy_data = np.random.rand(2, 2).astype(np.float32)
    data = np.random.rand(bps, 2, 2).astype(np.float32)
    weights = np.random.rand(2, 2).astype(np.float32)

    vgraph_ids = []
    ps_ids = []

    def init_builder(builder):
        d0 = builder.addInputTensor(dummy_data, 'data0')
        w0 = builder.addInitializedInputTensor(weights)

        mm0 = builder.aiOnnx.matmul([d0, w0], "mm0")
        s0 = builder.aiOnnx.sin([mm0])
        mm1 = builder.aiOnnx.matmul([s0, w0], "mm1")

        builder.addOutputTensor(mm1)

        builder.pipelineStage(mm0, 0)
        builder.pipelineStage(s0, 1)
        builder.pipelineStage(mm1, 2)

        builder.virtualGraph(mm0, 0)
        builder.virtualGraph(s0, 1)
        builder.virtualGraph(mm1, 0)

        loss = builder.addL1Loss(mm1, 'l1LossVal', 0.1,
                                 popart.ReductionType.Sum)
        loss.virtualGraph(0)
        loss.pipelineStage(2)

        return [mm1]

    def ref():
        mm0 = np.matmul(data, weights)
        s0 = np.sin(mm0)
        mm1 = np.matmul(s0, weights)
        return mm1

    session = PopartTestSession()
    session.options.enableVirtualGraphs = True
    session.options.enablePipelining = True
    session.device = 'ipu_model'
    session.numIPUs = 2
    session.batchesPerStep = bps

    # test a pipeline stage appearing on multiple virtual graphs
    session.prepare(init_builder)
    sessionAnchors = session.run({'data0': data})
    assert len(sessionAnchors) == 1
    sessionAnchors = [v for k, v in sessionAnchors.items()][0]
    print(sessionAnchors)

    print()

    refAnchors = ref()
    print(refAnchors)

    assert np.allclose(sessionAnchors, refAnchors)


# run the same model with and without revisiting ipus and compare the resultant weights.
def test_multiple_stages_per_virtual_graph_training():
    accumulation_factor = 5
    micro_batches_per_step = 5
    bps = micro_batches_per_step // accumulation_factor
    dummy_data = np.random.rand(2, 2).astype(np.float32)
    data = np.random.rand(accumulation_factor, 2, 2).astype(np.float32)
    weight_data = np.random.rand(2, 2).astype(np.float32)

    def run_test(set_pipeline_stages):
        weights = {}

        def init_builder(builder):
            d0 = builder.addInputTensor(dummy_data, 'data0')
            w0 = builder.addInitializedInputTensor(weight_data)
            weights[w0] = np.empty(shape=weight_data.shape,
                                   dtype=weight_data.dtype)

            t0 = builder.aiOnnx.matmul([d0, w0])
            t1 = builder.aiOnnx.sin([t0])
            t2 = builder.aiOnnx.matmul([t1, w0])

            builder.addOutputTensor(t2)

            if set_pipeline_stages:
                builder.pipelineStage(t0, 0)
                builder.pipelineStage(t1, 1)
                builder.pipelineStage(t2, 2)

                builder.virtualGraph(t0, 0)
                builder.virtualGraph(t1, 1)
                builder.virtualGraph(t2, 0)

            loss = builder.addL1Loss(t2, 'l1LossVal', 0.1,
                                     popart.ReductionType.Sum)
            if set_pipeline_stages:
                loss.pipelineStage(2)
                loss.virtualGraph(0)

            return [t2]

        session = PopartTestSession()
        session.mode = 'train'
        session.options.enableVirtualGraphs = set_pipeline_stages
        session.options.enablePipelining = set_pipeline_stages
        session.device = 'ipu_model'
        if set_pipeline_stages:
            session.numIPUs = 2
        session.batchesPerStep = bps
        session.options.enableGradientAccumulation = True
        session.options.accumulationFactor = accumulation_factor

        # test a pipeline stage appearing on multiple virtual graphs
        session.prepare(init_builder)

        sessionAnchors = session.run({'data0': data})
        assert len(sessionAnchors) == 1
        sessionAnchor = [v for k, v in sessionAnchors.items()][0]

        session._session.weightsToHost()
        weightsIo = popart.PyWeightsIO(weights)
        session._session.readWeights(weightsIo)
        assert len(weights) == 1
        weights = [v for k, v in weights.items()]
        return weights[0], sessionAnchor

    w0, r0 = run_test(False)
    w1, r1 = run_test(True)

    print("Single Ipu with gradient accumulation:")
    print("  Result:")
    print(f"    {r0}")
    print("  Weights:")
    print(f"    {w0}")
    print()
    print("Pipelining with multiple stages per ipu:")
    print("  Result:")
    print(f"    {r1}")
    print("  Weights:")
    print(f"    {w1}")

    assert np.allclose(r0, r1)
    assert np.allclose(w0, w1)


# run the same model with and without recomputation and check the updated weights
def test_recomputation():
    accumulationFactor = 3
    microBatchesPerStep = 3
    bps = microBatchesPerStep // accumulationFactor
    dummy_data = np.zeros((2, 2)).astype(np.float32)
    data = np.array([i for i in range(accumulationFactor * 2 * 2)]).astype(
        np.float32) * 0.1
    data = np.reshape(data, (accumulationFactor, 2, 2))
    weight_data = np.array([i for i in range(2 * 2)]).astype(np.float32) * 0.25
    weight_data = np.reshape(weight_data, (2, 2))

    def run_test(enable_recomputation):
        weights = {}

        def init_builder(builder):
            d0 = builder.addInputTensor(dummy_data, 'data0')
            w0 = builder.addInitializedInputTensor(weight_data)
            weights[w0] = np.empty(shape=weight_data.shape,
                                   dtype=weight_data.dtype)

            t0 = builder.aiOnnx.mul([d0, w0])
            t1 = builder.aiOnnx.sigmoid([t0])
            t2 = builder.aiGraphcore.scale([t1], 2.0)

            for t in (t0, t1, t2):
                builder.virtualGraph(t, 0)

            loss = builder.addL1Loss(t2, 'l1LossVal', 0.159,
                                     popart.ReductionType.Sum)
            loss.virtualGraph(1)

            return [t2]

        session = PopartTestSession()
        session.device = 'ipu_model'
        session.numIPUs = 2
        session.mode = 'train'
        session.options.virtualGraphMode = popart.VirtualGraphMode.Manual
        session.options.enablePipelining = True
        if enable_recomputation:
            session.options.autoRecomputation = popart.RecomputationType.Standard
        session.options.accumulationFactor = accumulationFactor
        session.options.enableGradientAccumulation = True

        session.prepare(init_builder)

        anchors = session.run({'data0': data})

        # return the weights
        session._session.weightsToHost()
        weightsIo = popart.PyWeightsIO(weights)
        session._session.readWeights(weightsIo)
        assert len(weights) == 1
        weights = [v for k, v in weights.items()]
        return weights[0]

    w0 = run_test(False)
    w1 = run_test(True)

    print(w0)
    print()
    print(w1)
    print()

    diff = w0 - w1
    print(diff)

    assert np.array_equal(w0, w1)


def test_bad_auto_staging():
    bps = 4
    dummy_data = np.random.rand(2, 2).astype(np.float32)
    data = np.random.rand(bps, 2, 2).astype(np.float32)

    vgraph_ids = []
    ps_ids = []

    def init_builder(builder):
        d0 = builder.addInputTensor(dummy_data, 'data0')

        t0 = builder.aiOnnx.sin([d0])
        t1 = builder.aiOnnx.sin([t0])
        t2 = builder.aiOnnx.sin([t1])

        builder.addOutputTensor(t2)

        builder.virtualGraph(t0, 0)
        builder.virtualGraph(t1, 1)
        builder.virtualGraph(t2, 0)

        loss = builder.addL1Loss(t2, 'l1LossVal', 0.1,
                                 popart.ReductionType.Sum)
        loss.virtualGraph(0)

        return [t2]

    def ref(d0):
        t0 = np.sin(d0)
        t1 = np.sin(t0)
        t2 = np.sin(t1)
        return t2

    session = PopartTestSession()
    session.options.enableVirtualGraphs = True
    session.options.enablePipelining = True
    session.device = 'ipu_model'
    session.numIPUs = 2
    session.batchesPerStep = bps

    # test a pipeline stage appearing on multiple virtual graphs
    with pytest.raises(popart.popart_exception) as e_info:
        session.prepare(init_builder)

    assert e_info.value.args[0].startswith(
        'Tensor Sin:0/1 is consumed in an earlier pipeline stage than it is produced'
    )

    # The below lines should be uncommented when auto pipeline stage is improved.
    # assert len(sessionAnchors) == 1
    # result = [v for k, v in sessionAnchors.items()][0]

    # for i in range(bps):
    #     refResult = ref(data[i])
    #     print(f'Batch {i}: {result[i]}')
    #     print(f'Ref result: {refResult}')
    #     print()

    #     assert np.allclose(result[i], refResult)
