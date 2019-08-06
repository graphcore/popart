import numpy as np
import pytest
import popart


def test_disabled_virtual_graphs():
    """
    In this test we check that an error is thrown when doing pipelining
    if enableVirtualGraph session option is not set to true
    """
    builder, op0_out, op1_out, op2_out, op3_out, anchor_map, loss = get_simple_linear_model(
    )

    opts = popart.SessionOptionsCore()
    opts.enablePipelining = True
    opts.enableVirtualGraphs = False

    with pytest.raises(popart.popart_exception) as e_info:
        session = popart.InferenceSession(
            fnModel=builder.getModelProto(),
            dataFeed=popart.DataFlow(10, anchor_map),
            userOptions=opts,
            losses=[loss],
            deviceInfo=popart.DeviceManager().createIpuModelDevice({}))
    assert e_info.value.args[0].startswith(
        "Pipelining requires the 'enableVirtualGraphs' session option")


def test_enabled_recomputation():
    """
    In this test we check that an error is thrown when doing pipelining
    if recomputation is enabled
    """
    builder, op0_out, op1_out, op2_out, op3_out, anchor_map, loss = get_simple_linear_model(
    )

    opts = popart.SessionOptionsCore()
    opts.enablePipelining = True
    opts.enableVirtualGraphs = True
    opts.autoRecomputation = popart.RecomputationType.Standard

    builder.virtualGraph(op0_out, 0)
    builder.virtualGraph(op1_out, 1)
    builder.virtualGraph(op2_out, 1)
    builder.virtualGraph(op3_out, 1)

    with pytest.raises(popart.popart_exception) as e_info:
        session = popart.InferenceSession(
            fnModel=builder.getModelProto(),
            dataFeed=popart.DataFlow(10, anchor_map),
            userOptions=opts,
            losses=[loss],
            deviceInfo=popart.DeviceManager().createIpuModelDevice({
                'numIPUs':
                2,
                "tilesPerIPU":
                20
            }))
    assert e_info.value.args[0].startswith(
        "Auto recomputation for Pipelining needs implementing")


def test_bad_sharding0():
    """
    Non-linear sharding throws error.
    For our graph : Op0 -> Op1 -> Op2 -> Op3 -> Loss
    consider the three cases
      1) IPU0 : {Op2, Op3}, IPU1 : {Op0, Op1, Loss}
      2) IPU0 : {Op0, Op2}, IPU1 : {Op1, Op3, Loss}
      3) IPU0 : {Op0, Op1, Loss}, IPU1 : {Op2, Op3}
    """

    opts = popart.SessionOptionsCore()
    opts.enablePipelining = True
    opts.enableVirtualGraphs = True

    # 1)
    builder, op0_out, op1_out, op2_out, op3_out, anchor_map, loss = get_simple_linear_model(
    )
    builder.virtualGraph(op0_out, 1)
    builder.virtualGraph(op1_out, 1)
    builder.virtualGraph(op2_out, 0)
    builder.virtualGraph(op3_out, 0)
    loss.virtualGraph(1)

    with pytest.raises(popart.popart_exception) as e_info:
        session = popart.InferenceSession(
            fnModel=builder.getModelProto(),
            dataFeed=popart.DataFlow(10, anchor_map),
            userOptions=opts,
            losses=[loss],
            deviceInfo=popart.DeviceManager().createIpuModelDevice({
                'numIPUs':
                2,
                "tilesPerIPU":
                20
            }))
    assert e_info.value.args[0].find("forward IPU copies go from IPU N to N+1")

    # 2)
    builder, op0_out, op1_out, op2_out, op3_out, anchor_map, loss = get_simple_linear_model(
    )
    builder.virtualGraph(op0_out, 0)
    builder.virtualGraph(op1_out, 1)
    builder.virtualGraph(op2_out, 0)
    builder.virtualGraph(op3_out, 1)
    loss.virtualGraph(1)

    with pytest.raises(popart.popart_exception) as e_info:
        session = popart.InferenceSession(
            fnModel=builder.getModelProto(),
            dataFeed=popart.DataFlow(10, anchor_map),
            userOptions=opts,
            losses=[loss],
            deviceInfo=popart.DeviceManager().createIpuModelDevice({
                'numIPUs':
                2,
                "tilesPerIPU":
                20
            }))
    assert e_info.value.args[0].find("forward IPU copies go from IPU N to N+1")

    # 3)
    builder, op0_out, op1_out, op2_out, op3_out, anchor_map, loss = get_simple_linear_model(
    )
    builder.virtualGraph(op0_out, 0)
    builder.virtualGraph(op1_out, 0)
    builder.virtualGraph(op2_out, 1)
    builder.virtualGraph(op3_out, 1)
    loss.virtualGraph(0)

    with pytest.raises(popart.popart_exception) as e_info:
        session = popart.InferenceSession(
            fnModel=builder.getModelProto(),
            dataFeed=popart.DataFlow(10, anchor_map),
            userOptions=opts,
            losses=[loss],
            deviceInfo=popart.DeviceManager().createIpuModelDevice({
                'numIPUs':
                2,
                "tilesPerIPU":
                20
            }))
    assert e_info.value.args[0].find(
        "such that the loss is on the final IPU in the pipeline")


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

    opts = popart.SessionOptionsCore()
    opts.enablePipelining = True
    opts.enableVirtualGraphs = True

    builder.virtualGraph(op0_out, 0)
    builder.virtualGraph(op1_out, 1)
    builder.virtualGraph(op2_out, 1)
    builder.virtualGraph(op3_out, 1)
    loss.virtualGraph(1)

    session = popart.InferenceSession(
        fnModel=builder.getModelProto(),
        dataFeed=popart.DataFlow(10, anchor_map),
        userOptions=opts,
        losses=[loss],
        deviceInfo=popart.DeviceManager().createIpuModelDevice({
            'numIPUs':
            2,
            "tilesPerIPU":
            20
        }))


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

    art = popart.AnchorReturnType("ALL")
    loss = popart.NllLoss(op2_out, l0, "loss")
    anchor_map = {op2_out: art, "loss": art}

    opts = popart.SessionOptionsCore()
    opts.enablePipelining = True
    opts.enableVirtualGraphs = True

    builder.virtualGraph(op0_out, 0)
    builder.virtualGraph(op1_out, 1)
    builder.virtualGraph(op2_out, 2)
    loss.virtualGraph(2)

    session = popart.InferenceSession(
        fnModel=builder.getModelProto(),
        dataFeed=popart.DataFlow(10, anchor_map),
        userOptions=opts,
        losses=[loss],
        deviceInfo=popart.DeviceManager().createIpuModelDevice({
            'numIPUs':
            3,
            "tilesPerIPU":
            20
        }))


def test_inference_min_batches():
    """
    Check that we throw if too few batches to fill and flush the pipeline
    for an inference model
    """
    minBatches = 3  # numIpus

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
    minBatches = 5  # 2 * (numIpus-1) + 1

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
    def test(do_pipelining):
        dsize = 10
        ratio = 0.5
        if do_pipelining:
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
            if do_pipelining:
                builder.virtualGraph(identity0, vgraph_num)

            [dropout0] = builder.aiOnnx.dropout([identity0],
                                                num_outputs=1,
                                                ratio=ratio)
            if do_pipelining:
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
            if do_pipelining:
                builder.virtualGraph(add0, vgraph_num)

            return add0

        # construct a graph of `layers` number of layers
        # with each layer on a different IPU.
        next_layer_in = ip
        for vgraph in range(layers):
            next_layer_in = add_layer(next_layer_in, vgraph)
        out = next_layer_in
        builder.addOutputTensor(out)

        device = popart.DeviceManager().acquireAvailableDevice(numIpus=ipus)
        if device is None:
            pytest.skip("Test needs to run on IPU, but none are available")

        dfAnchors = {}
        for x in dropouts + dropoutGrads + dropoutInputs:
            dfAnchors[x] = popart.AnchorReturnType("ALL")

        dataFlow = popart.DataFlow(batches_per_step, dfAnchors)

        loss = popart.L1Loss(out, "l1LossVal", 0.1)
        if do_pipelining:
            loss.virtualGraph(layers - 1)

        userOptions = popart.SessionOptions()
        userOptions.enableVirtualGraphs = do_pipelining
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

    # Test without pipelining for debugging purposes.
    # test(do_pipelining=False)

    test(do_pipelining=True)


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

    art = popart.AnchorReturnType("ALL")
    loss = popart.NllLoss(out, l0, "loss")

    anchor_map = {"loss": art, w0: art, e0: art}
    if doTraining is True:
        anchor_map[popart.reservedGradientPrefix() + d0] = art
        if doPipelining is True and anchorRestoredTensors is True:
            anchor_map[popart.reservedRestoredPrefix() + e0] = art
            anchor_map[d0] = art
            anchor_map[popart.reservedRestoredPrefix() + d0] = art

    opts = popart.SessionOptionsCore()
    opts.reportOptions = {"showExecutionSteps": "true"}
    opts.enablePipelining = doPipelining

    if doSharding is False:
        deviceOpts = {'numIPUs': 1, "tilesPerIPU": 20}
    else:
        opts.enableVirtualGraphs = True
        deviceOpts = {'numIPUs': 3, "tilesPerIPU": 20}
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
            deviceInfo=popart.DeviceManager().createIpuModelDevice(deviceOpts))
    else:
        session = popart.InferenceSession(
            fnModel=builder.getModelProto(),
            dataFeed=popart.DataFlow(batchesPerStep, anchor_map),
            losses=[loss],
            userOptions=opts,
            deviceInfo=popart.DeviceManager().createIpuModelDevice(deviceOpts))

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

    art = popart.AnchorReturnType("ALL")
    loss = popart.NllLoss(op3_out, l0, "loss")
    anchor_map = {op3_out: art, "loss": art}

    return builder, op0_out, op1_out, op2_out, op3_out, anchor_map, loss
