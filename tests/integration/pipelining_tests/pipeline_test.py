# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
from collections import namedtuple
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
    builder, _, _, _, _, anchor_map = get_simple_linear_model()

    opts = popart.SessionOptions()
    opts.enablePipelining = True
    opts.virtualGraphMode = popart.VirtualGraphMode.Off

    with tu.create_test_device() as device:
        with pytest.raises(popart.popart_exception) as e_info:
            _ = popart.InferenceSession(
                fnModel=builder.getModelProto(),
                dataFlow=popart.DataFlow(10, anchor_map),
                userOptions=opts,
                deviceInfo=device,
            )
        assert e_info.value.args[0].startswith("Pipelining requires more than")


@tu.requires_ipu_model
def test_one_ipu():
    """
    In this test we check that an error is thrown when doing pipelining
    on 1 IPU
    """
    builder = popart.Builder()
    shape_d = [10]
    d0 = builder.addInputTensor(popart.TensorInfo("FLOAT", shape_d))
    d1 = builder.addInputTensor(popart.TensorInfo("FLOAT", shape_d))
    op0_out = builder.aiOnnx.sin([d0], "s0")
    op1_out = builder.aiOnnx.exp([d1], "r0")
    op2_out = builder.aiOnnx.mul([op0_out, op1_out], "m0")
    builder.addOutputTensor(op2_out)
    opts = popart.SessionOptions()
    opts.enablePipelining = True
    opts.virtualGraphMode = popart.VirtualGraphMode.Manual  # i.e. use 1 ipu
    builder.pipelineStage(op0_out, 0)
    builder.virtualGraph(op0_out, 0)
    builder.pipelineStage(op1_out, 0)
    builder.virtualGraph(op1_out, 0)
    builder.pipelineStage(op2_out, 1)
    builder.virtualGraph(op2_out, 0)

    with tu.create_test_device() as device:
        with pytest.raises(popart.popart_exception) as e_info:
            session = popart.InferenceSession(
                fnModel=builder.getModelProto(),
                dataFlow=popart.DataFlow(10, [op2_out, "loss"]),
                userOptions=opts,
                deviceInfo=device,
            )
            session.prepareDevice()
        assert e_info.value.args[0].startswith("Pipelining requires more than")


@tu.requires_ipu_model
@pytest.mark.parametrize("explicit", [False, True])
def test_enabled_recomputation(explicit):
    """
    In this test we check that NO error is thrown when doing pipelining
    if recomputation is enabled
    """
    builder, op0_out, op1_out, op2_out, op3_out, anchor_map = get_simple_linear_model()

    opts = popart.SessionOptions()
    opts.enablePipelining = True
    opts.virtualGraphMode = popart.VirtualGraphMode.Manual
    opts.autoRecomputation = popart.RecomputationType.Standard
    opts.enableExplicitIR(explicit)

    builder.virtualGraph(op0_out, 0)
    builder.virtualGraph(op1_out, 1)
    builder.virtualGraph(op2_out, 1)
    builder.virtualGraph(op3_out, 1)

    with tu.create_test_device(numIpus=2, tilesPerIPU=20) as device:
        _ = popart.InferenceSession(
            fnModel=builder.getModelProto(),
            dataFlow=popart.DataFlow(10, anchor_map),
            userOptions=opts,
            deviceInfo=device,
        )


@tu.requires_ipu_model
@pytest.mark.parametrize("explicit", [False, True])
def test_stream_tensors_to_multiple_ipus(explicit):
    """
    Streaming an input to Ops on multiple IPUs throws an error

    09/07/2019 Since D12445 this test no longer raises an exception. By
    default, stream tensors are now replicated by streaming to a single
    IPU, then copied across to the other IPUs where they are needed.
    Leaving this test in to verify that this remains the case
    """
    builder, op0_out, op1_out, op2_out, op3_out, anchor_map = get_simple_linear_model(
        streamInputToOp1AndOp2=True
    )

    opts = popart.SessionOptions()
    opts.enablePipelining = True
    opts.virtualGraphMode = popart.VirtualGraphMode.Manual
    opts.enableExplicitIR(explicit)

    builder.virtualGraph(op0_out, 0)
    builder.virtualGraph(op1_out, 1)
    builder.virtualGraph(op2_out, 1)
    builder.virtualGraph(op3_out, 1)

    with tu.create_test_device(numIpus=2, tilesPerIPU=20) as device:
        _ = popart.InferenceSession(
            fnModel=builder.getModelProto(),
            dataFlow=popart.DataFlow(10, anchor_map),
            userOptions=opts,
            deviceInfo=device,
        )


@tu.requires_ipu_model
@pytest.mark.parametrize("explicit", [False, True])
def test_sharding_multi_source(explicit):
    """
    Branched sharding does not merge IPU Copies with pipelining
    e.g. Op0 -> Op2
                 ^
         Op1 ----'
    where the vGraph split is IPU0 : {Op0}, IPU1 : {Op1}, IPU2 : {Op2}
    """
    builder = popart.Builder()
    shape_d = [10]
    shape_l = []
    d0 = builder.addInputTensor(popart.TensorInfo("FLOAT", shape_d))
    d1 = builder.addInputTensor(popart.TensorInfo("FLOAT", shape_d))
    l0 = builder.addInputTensor(popart.TensorInfo("INT32", shape_l))

    op0_out = builder.aiOnnx.sin([d0], "s0")
    op1_out = builder.aiOnnx.exp([d1], "r0")
    op2_out = builder.aiOnnx.mul([op0_out, op1_out], "m0")
    nll = builder.aiGraphcore.nllloss([op2_out, l0])

    opts = popart.SessionOptions()
    opts.enablePipelining = True
    opts.virtualGraphMode = popart.VirtualGraphMode.Manual
    opts.enableExplicitIR(explicit)

    builder.virtualGraph(op0_out, 0)
    builder.virtualGraph(op1_out, 1)
    builder.virtualGraph(op2_out, 2)
    builder.virtualGraph(nll, 2)

    with tu.create_test_device(numIpus=3, tilesPerIPU=20) as device:
        _ = popart.InferenceSession(
            fnModel=builder.getModelProto(),
            dataFlow=popart.DataFlow(10, [op2_out]),
            userOptions=opts,
            deviceInfo=device,
        )


@tu.requires_ipu_model
def test_inference_min_batches():
    """
    Check that we throw if too few batches to fill and flush the pipeline
    for an inference model
    """
    minBatches = 3  # == numIpus == numPipelineStages

    get_model_anchors(
        doSharding=True,
        doPipelining=True,
        batchesPerStep=minBatches,
        doTraining=False,
        doDevicex=False,
    )

    with pytest.raises(popart.popart_exception) as e_info:
        get_model_anchors(
            doSharding=True,
            doPipelining=True,
            batchesPerStep=minBatches - 1,
            doTraining=False,
            doDevicex=False,
        )
    assert e_info.value.args[0].startswith(
        "For pipelining, depth (batchesPerStep) must"
    )


@tu.requires_ipu_model
def test_training_min_batches():
    """
    Check that we throw if too few batches to fill and flush the pipeline
    for a training model
    """
    minBatches = 5  # == 2 * (numIpus-1) + 1 == numPipelineStages

    get_model_anchors(
        doSharding=True,
        doPipelining=True,
        batchesPerStep=minBatches,
        doTraining=True,
        doDevicex=False,
    )

    with pytest.raises(popart.popart_exception) as e_info:
        get_model_anchors(
            doSharding=True,
            doPipelining=True,
            batchesPerStep=minBatches - 1,
            doTraining=True,
            doDevicex=False,
        )
    assert e_info.value.args[0].startswith(
        "For pipelining, depth (batchesPerStep) must"
    )


_DataType = namedtuple("_DataType", ["builder_type", "np_type"])
_INT8 = _DataType("INT8", np.int8)
_UINT8 = _DataType("UINT8", np.uint8)


@tu.requires_ipu_model
@pytest.mark.parametrize("inputType", [_INT8, _UINT8, None])
@pytest.mark.parametrize("explicit", [False, True])
def test_output_matches_train(inputType, explicit):
    """
    In this test we check that the anchors of equivalent non-sharded, sharded
    and non-pipelined, and sharded and pipelined models are equal when doing
    training. We expect only the first output and weight update to be the same
    as non-pipelined models
    """
    bps = 8
    singleIpu_anchors = get_model_anchors(
        doSharding=False,
        doPipelining=False,
        batchesPerStep=bps,
        doTraining=True,
        inputType=inputType,
        explicit=explicit,
    )
    multiIpu_anchors = get_model_anchors(
        doSharding=True,
        doPipelining=False,
        batchesPerStep=bps,
        doTraining=True,
        inputType=inputType,
        explicit=explicit,
    )
    pipelined_anchors = get_model_anchors(
        doSharding=True,
        doPipelining=True,
        batchesPerStep=bps,
        doTraining=True,
        inputType=inputType,
        explicit=explicit,
    )

    for (tId1, t1), (tId2, t2) in zip(
        singleIpu_anchors.items(), multiIpu_anchors.items()
    ):
        assert np.allclose(t1, t2)

    # Expect only the anchors from the first batch to be equal. After that, the
    # continuous gradient accumulation option causes model parameters to diverge
    for (tId1, t1), (tId2, t2) in zip(
        singleIpu_anchors.items(), pipelined_anchors.items()
    ):
        for i in range(np.shape(t1)[0]):
            print("singleIpu   , batch: ", i, tId1, np.sum(t1[i]))
            print("pipelinedIpu, batch: ", i, tId2, np.sum(t2[i]))
        assert np.allclose(t1[0], t2[0])


@tu.requires_ipu_model
@pytest.mark.parametrize("inputType", [_INT8, _UINT8, None])
@pytest.mark.parametrize("explicit", [False])
def test_acts_match_restored_acts(inputType, explicit):
    """
    In this test we check that the stashed tensors and their equivalent
    Restored tensors have the same values for all batches. This confirms
    that the schedule of restoring and streaming anchors is correct

    How do we know they're not both wrong? Take this example where the
    streamed input is stashed. Check that it matches the raw data input
    that is fed to the StepIO
    """
    bps = 8
    pipelined_anchors = get_model_anchors(
        doSharding=True,
        doPipelining=True,
        batchesPerStep=bps,
        doTraining=True,
        anchorRestoredTensors=True,
        returnRawInput=True,
        inputType=inputType,
        explicit=explicit,
    )

    for (tId, t) in pipelined_anchors.items():
        for i in range(np.shape(t)[0]):
            print("batch: ", i, tId, np.sum(t[i]))

    # Can't seem to make the cast op produce a tensor with id "input", so we
    # have to do this instead.
    input_name = "Cast:0" if inputType is not None else "input"

    assert np.allclose(
        pipelined_anchors[popart.reservedRestoredPrefix() + "Exp:0"],
        pipelined_anchors["Exp:0"],
    )
    assert np.allclose(
        pipelined_anchors[popart.reservedRestoredPrefix() + input_name],
        pipelined_anchors[input_name],
    )
    assert np.allclose(pipelined_anchors["input_raw"], pipelined_anchors[input_name])


@tu.requires_ipu_model
@pytest.mark.parametrize("inputType", [_INT8, _UINT8, None])
@pytest.mark.parametrize("explicit", [False, True])
def test_output_matches_infer(inputType, explicit):
    """
    In this test we check that the anchors of equivalent non-sharded, sharded
    and non-pipelined, and sharded and pipelined models are equal when doing
    inference
    """
    bps = 8
    singleIpu_anchors = get_model_anchors(
        doSharding=False,
        doPipelining=False,
        batchesPerStep=bps,
        doTraining=False,
        inputType=inputType,
        explicit=explicit,
    )
    multiIpu_anchors = get_model_anchors(
        doSharding=True,
        doPipelining=False,
        batchesPerStep=bps,
        doTraining=False,
        inputType=inputType,
        explicit=explicit,
    )
    pipelined_anchors = get_model_anchors(
        doSharding=True,
        doPipelining=True,
        batchesPerStep=bps,
        doTraining=False,
        inputType=inputType,
        explicit=explicit,
    )

    for (tId1, t1), (tId2, t2) in zip(
        singleIpu_anchors.items(), multiIpu_anchors.items()
    ):
        for i in range(np.shape(t1)[0]):
            print("singleIpu, batch: ", i, tId1, np.sum(t1[i]))
            print("multiIpu , batch: ", i, tId2, np.sum(t2[i]))
        assert np.allclose(t1, t2)

    for (tId1, t1), (tId2, t2) in zip(
        singleIpu_anchors.items(), pipelined_anchors.items()
    ):
        for i in range(np.shape(t1)[0]):
            print("singleIpu   , batch: ", i, tId1, np.sum(t1[i]))
            print("pipelinedIpu, batch: ", i, tId2, np.sum(t2[i]))
        assert np.allclose(t1, t2)


# Model
#  <--- ipu0 ----> <--------- ipu1 ---> <------------ ipu2 ------------>
#
#  d0 --|-- Sin --|-- Exp --|
#                           |-- Conv --|-- Reshape --|-- Softmax --> out
#                      w0 --|
def get_model_anchors(
    doSharding,
    doPipelining,
    batchesPerStep,
    doTraining,
    doProfiling=False,
    doDevicex=True,
    anchorRestoredTensors=False,
    returnRawInput=False,
    inputType=None,
    explicit=False,
):
    np.random.seed(seed=1)

    builder = popart.Builder()
    batchSize = 2
    shape_d0 = [batchSize, 2, 4, 4]
    shape_l0 = [batchSize]
    if inputType is not None:
        d0 = builder.addInputTensor(popart.TensorInfo(inputType.builder_type, shape_d0))
    else:
        d0 = builder.addInputTensor(popart.TensorInfo("FLOAT", shape_d0))
    data_w0 = np.ones(shape=[2, 2, 3, 3]).astype(np.float32)
    w0 = builder.addInitializedInputTensor(data_w0)
    l0 = builder.addInputTensor(popart.TensorInfo("INT32", shape_l0))

    if inputType is not None:
        d0_cast = builder.aiOnnx.cast([d0], "FLOAT")
    else:
        d0_cast = d0

    s0 = builder.aiOnnx.sin([d0_cast], "s0")
    e0 = builder.aiOnnx.exp([s0], "e0")
    c0 = builder.aiOnnx.conv(
        [e0, w0], dilations=[1, 1], pads=[1, 1, 1, 1], strides=[1, 1], debugContext="c0"
    )
    r0 = builder.reshape_const(builder.aiOnnx, [c0], [batchSize, 32])
    out = builder.aiOnnx.softmax([r0], axis=1, debugContext="sfm")
    nll = builder.aiGraphcore.nllloss([out, l0])

    art = popart.AnchorReturnType("All")

    anchor_map = {nll: art, w0: art, e0: art}
    if doTraining is True:
        anchor_map[popart.reservedGradientPrefix() + d0_cast] = art
        if doPipelining is True and anchorRestoredTensors is True:
            anchor_map[popart.reservedRestoredPrefix() + e0] = art
            anchor_map[d0_cast] = art
            anchor_map[popart.reservedRestoredPrefix() + d0_cast] = art

    opts = popart.SessionOptions()
    opts.reportOptions = {"showExecutionSteps": "true"}
    opts.enablePipelining = doPipelining
    opts.enableExplicitIR(explicit)

    if doSharding is False:
        numIPUs = 1
    else:
        opts.virtualGraphMode = popart.VirtualGraphMode.Manual
        numIPUs = 3
        if inputType is not None:
            builder.virtualGraph(d0_cast, 0)
        builder.virtualGraph(s0, 0)
        builder.virtualGraph(e0, 1)
        builder.virtualGraph(c0, 1)
        builder.virtualGraph(r0, 2)
        builder.virtualGraph(out, 2)
        builder.virtualGraph(nll, 2)

    with tu.create_test_device(numIpus=numIPUs, tilesPerIPU=20) as device:
        if doTraining is True:
            session = popart.TrainingSession(
                fnModel=builder.getModelProto(),
                dataFlow=popart.DataFlow(batchesPerStep, anchor_map),
                loss=nll,
                optimizer=popart.ConstSGD(0.01),
                userOptions=opts,
                deviceInfo=device,
            )
        else:
            session = popart.InferenceSession(
                fnModel=builder.getModelProto(),
                dataFlow=popart.DataFlow(batchesPerStep, anchor_map),
                userOptions=opts,
                deviceInfo=device,
            )

        if doDevicex is False:
            return None

        anchors = session.initAnchorArrays()
        session.prepareDevice()

        if batchesPerStep > 1:
            shape_d0.insert(0, batchesPerStep)
            shape_l0.insert(0, batchesPerStep)
        d0_host_type = inputType.np_type if inputType is not None else np.float32
        data = np.random.uniform(low=-10.0, high=10.0, size=shape_d0).astype(
            d0_host_type
        )
        classes = np.prod(shape_d0) / (batchSize * batchesPerStep)
        label = np.random.randint(low=0, high=classes, size=shape_l0).astype(np.int32)

        inputs = {d0: data, l0: label}
        stepio = popart.PyStepIO(inputs, anchors)

        session.weightsFromHost()

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

    d0 = builder.addInputTensor(popart.TensorInfo("FLOAT", shape_d))
    d1 = builder.addInputTensor(popart.TensorInfo("FLOAT", shape_d))

    op0_out = builder.aiOnnx.sin([d0], "s0")
    if streamInputToOp1AndOp2 is True:
        op1_out = builder.aiOnnx.mul([op0_out, d0])
    else:
        op1_out = builder.aiOnnx.mul([op0_out, d1])
    op2_out = builder.aiOnnx.exp([op1_out], "e0")
    op3_out = builder.aiOnnx.exp([op2_out], "e1")
    builder.addOutputTensor(op3_out)

    art = popart.AnchorReturnType("All")
    anchor_map = {op3_out: art}

    return builder, op0_out, op1_out, op2_out, op3_out, anchor_map


@tu.requires_ipu_model
def test_pipeline_stage_errors():
    dummy_data = np.zeros(2, dtype=np.float32)
    bps = 2

    vgraph_ids = []
    ps_ids = []

    def init_builder(builder):
        d0 = builder.addInputTensor(dummy_data, "data0")
        _ = builder.addInputTensor(dummy_data, "data1")
        _ = builder.addInputTensor(dummy_data, "data2")

        s0 = builder.aiOnnx.sin([d0], "s0")
        m0 = builder.aiOnnx.mul([s0, d0])
        e0 = builder.aiOnnx.exp([m0])
        e1 = builder.aiOnnx.exp([e0])
        loss = builder.aiGraphcore.identityloss([e1])

        builder.addOutputTensor(loss)

        print(f"Setting virtual graphs to {vgraph_ids}")
        for tid, vgid in zip((s0, m0, e0, e1, loss), vgraph_ids):
            if vgid is not None:
                builder.virtualGraph(tid, vgid)

        print(f"Setting pipeline stages to {ps_ids}")
        for tid, psid in zip((s0, m0, e0, e1, loss), ps_ids):
            if psid is not None:
                builder.pipelineStage(tid, psid)

        return [loss]

    session = PopartTestSession()
    session.options.virtualGraphMode = popart.VirtualGraphMode.Manual
    session.options.enablePipelining = True
    session.batchesPerStep = bps

    # test a pipeline stage appearing on multiple virtual graphs
    vgraph_ids = [0, 0, 0, 1, 1]
    ps_ids = [0, 0, 1, 1, 1]
    with tu.create_test_device(numIpus=2) as device:
        with pytest.raises(popart.popart_exception) as e_info:
            session.prepare(init_builder, device=device)

    emsg = e_info.value.args[0]
    assert re.match("Ops .* have the same pipeline stage 1,.*", emsg) is not None

    # test not all ops having a pipeline stage set
    vgraph_ids = [0, 0, 1, 1, 1]
    ps_ids = [0, 0, None, 1, 1]
    with tu.create_test_device(numIpus=2) as device:
        with pytest.raises(popart.popart_exception) as e_info:
            session.prepare(init_builder, device=device)

    emsg = e_info.value.args[0]
    assert emsg.startswith("Only some ops have had their pipeline stage set.")


@tu.requires_ipu_model
@pytest.mark.parametrize("explicit", [False, True])
def test_pipeline_stages_backwards_through_ipus(explicit):
    dummy_data = np.array([0.5, 1.0], dtype=np.float32)
    bps = 2

    def init_builder(builder):
        d0 = builder.addInputTensor(dummy_data, "data0")

        s0 = builder.aiOnnx.sin([d0], "s0")
        m0 = builder.aiOnnx.mul([s0, d0])
        e0 = builder.aiOnnx.exp([m0])
        e1 = builder.aiOnnx.exp([e0], "output")
        loss = builder.aiGraphcore.identityloss([e1])
        builder.addOutputTensor(loss)

        stage0 = [s0, m0]
        stage1 = [e0, e1, loss]

        stage0_vgraph = 1
        stage1_vgraph = 0

        for tid in stage0:
            builder.virtualGraph(tid, stage0_vgraph)
            builder.pipelineStage(tid, 0)

        for tid in stage1:
            builder.virtualGraph(tid, stage1_vgraph)
            builder.pipelineStage(tid, 1)

        return [e1]

    def ref():
        d0 = dummy_data
        s0 = np.sin(d0)
        m0 = s0 * d0
        e0 = np.exp(m0)
        e1 = np.exp(e0)
        return e1

    session = PopartTestSession()
    session.options.virtualGraphMode = popart.VirtualGraphMode.Manual
    session.options.enablePipelining = True
    session.batchesPerStep = bps
    session.options.enableExplicitIR(explicit)

    # test a pipeline stage appearing on multiple virtual graphs
    with tu.create_test_device(numIpus=2) as device:
        session.prepare(init_builder, device=device)
        pipelineAnchors = session.run()

    assert len(pipelineAnchors) == 1
    pipelineAnchors = [v for k, v in pipelineAnchors.items()]
    pipelineAnchor = pipelineAnchors[0]

    print(pipelineAnchor)
    print(ref())

    assert np.allclose(pipelineAnchor[0], ref())


@tu.requires_ipu_model
@pytest.mark.parametrize("explicit", [False, True])
def test_multiple_stages_per_virtual_graph_inference(explicit):
    bps = 4
    dummy_data = np.random.rand(2, 2).astype(np.float32)
    data = np.random.rand(bps, 2, 2).astype(np.float32)
    weights = np.random.rand(2, 2).astype(np.float32)

    def init_builder(builder):
        d0 = builder.addInputTensor(dummy_data, "data0")
        w0 = builder.addInitializedInputTensor(weights)

        mm0 = builder.aiOnnx.matmul([d0, w0], "mm0")
        s0 = builder.aiOnnx.sin([mm0])
        mm1 = builder.aiOnnx.matmul([s0, w0], "mm1")
        loss = builder.aiGraphcore.identityloss([mm1])

        builder.addOutputTensor(loss)

        builder.pipelineStage(mm0, 0)
        builder.pipelineStage(s0, 1)
        builder.pipelineStage(mm1, 2)
        builder.pipelineStage(loss, 2)

        builder.virtualGraph(mm0, 0)
        builder.virtualGraph(s0, 1)
        builder.virtualGraph(mm1, 0)
        builder.virtualGraph(loss, 0)

        return [mm1]

    def ref():
        mm0 = np.matmul(data, weights)
        s0 = np.sin(mm0)
        mm1 = np.matmul(s0, weights)
        return mm1

    session = PopartTestSession()
    session.options.virtualGraphMode = popart.VirtualGraphMode.Manual
    session.options.enablePipelining = True
    session.batchesPerStep = bps
    session.options.enableExplicitIR(explicit)

    # test a pipeline stage appearing on multiple virtual graphs
    with tu.create_test_device(numIpus=2) as device:
        session.prepare(init_builder, device=device)
        sessionAnchors = session.run({"data0": data})
    assert len(sessionAnchors) == 1
    sessionAnchors = [v for k, v in sessionAnchors.items()][0]
    print(sessionAnchors)

    print()

    refAnchors = ref()
    print(refAnchors)

    assert np.allclose(sessionAnchors, refAnchors)


@tu.requires_ipu_model
def test_multiple_stages_per_virtual_graph_sharing_same_tensor_inference():
    bps = 4
    dummy_data = np.random.rand(2, 2).astype(np.float32)
    data = np.arange(bps * 2 * 2).reshape(bps, 2, 2).astype(np.float32)

    def init_builder(builder):
        x = builder.addInputTensor(dummy_data, "data0")

        a = builder.aiOnnx.add([x, x])
        b = builder.aiOnnx.mul([a, a])
        c = builder.aiOnnx.mul([b, b])
        rb = builder.reshape_const(builder.aiOnnx, [b], [4])
        rc = builder.reshape_const(builder.aiOnnx, [c], [4])
        d = builder.aiOnnx.mul([rb, rc])

        builder.addOutputTensor(d)

        builder.pipelineStage(a, 0)
        builder.pipelineStage(b, 1)
        builder.pipelineStage(c, 2)
        builder.pipelineStage(rb, 3)
        builder.pipelineStage(rc, 3)
        builder.pipelineStage(d, 3)

        builder.virtualGraph(a, 0)
        builder.virtualGraph(b, 1)
        builder.virtualGraph(c, 0)
        builder.virtualGraph(rb, 1)
        builder.virtualGraph(rc, 1)
        builder.virtualGraph(d, 1)

        return [d]

    def ref():
        a = data + data
        b = a * a
        c = b * b
        d = b.reshape(bps, -1) * c.reshape(bps, -1)

        return d

    session = PopartTestSession()
    session.options.virtualGraphMode = popart.VirtualGraphMode.Manual
    session.options.enablePipelining = True
    session.options.autoRecomputation = popart.RecomputationType.Pipeline
    session.options.stashAllTensorsInferencePipeline = True
    session.batchesPerStep = bps

    with tu.create_test_device(numIpus=2) as device:
        session.prepare(init_builder, device=device)
        sessionAnchors = session.run({"data0": data})
    assert len(sessionAnchors) == 1
    sessionAnchors = [v for _, v in sessionAnchors.items()][0]

    refAnchors = ref()

    assert np.allclose(sessionAnchors, refAnchors)


# run the same model with and without revisiting ipus and compare the resultant weights.
@tu.requires_ipu_model
@pytest.mark.parametrize("inputType", [_INT8, _UINT8, None])
@pytest.mark.parametrize("explicit", [False, True])
def test_multiple_stages_per_virtual_graph_training(inputType, explicit):
    accumulation_factor = 5
    micro_batches_per_step = 5
    bps = micro_batches_per_step // accumulation_factor
    data_type = inputType.np_type if inputType is not None else np.float32
    dummy_data = np.random.rand(2, 2).astype(data_type)
    data = np.random.rand(accumulation_factor, 2, 2).astype(data_type)
    weight_data = np.random.rand(2, 2).astype(np.float32)

    def run_test(set_pipeline_stages):
        weights = {}

        def init_builder(builder):
            d0 = builder.addInputTensor(dummy_data, "data0")
            w0 = builder.addInitializedInputTensor(weight_data)
            weights[w0] = np.empty(shape=weight_data.shape, dtype=weight_data.dtype)
            if inputType is not None:
                d0_float = builder.aiOnnx.cast([d0], "FLOAT")
                t0 = builder.aiOnnx.matmul([d0_float, w0])
            else:
                t0 = builder.aiOnnx.matmul([d0, w0])
            t1 = builder.aiOnnx.sin([t0])
            t2 = builder.aiOnnx.matmul([t1, w0])
            loss = builder.aiGraphcore.identityloss([t2])

            builder.addOutputTensor(loss)

            if set_pipeline_stages:
                if inputType is not None:
                    builder.pipelineStage(d0_float, 0)
                builder.pipelineStage(t0, 0)
                builder.pipelineStage(t1, 1)
                builder.pipelineStage(t2, 2)
                builder.pipelineStage(loss, 2)

                if inputType is not None:
                    builder.virtualGraph(d0_float, 0)
                builder.virtualGraph(t0, 0)
                builder.virtualGraph(t1, 1)
                builder.virtualGraph(t2, 0)
                builder.virtualGraph(loss, 0)

            return [loss]

        session = PopartTestSession()
        session.mode = "train"
        session.options.enablePipelining = set_pipeline_stages
        numIpus = 1
        if set_pipeline_stages:
            numIpus = 2
            session.options.virtualGraphMode = popart.VirtualGraphMode.Manual
        session.batchesPerStep = bps
        session.options.enableGradientAccumulation = True
        session.options.accumulationFactor = accumulation_factor
        session.options.enableExplicitIR(explicit)

        # test a pipeline stage appearing on multiple virtual graphs
        with tu.create_test_device(numIpus=numIpus) as device:
            session.prepare(init_builder, device=device)

            sessionAnchors = session.run({"data0": data})
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
@tu.requires_ipu_model
@pytest.mark.parametrize("inputType", [_INT8, _UINT8, None])
@pytest.mark.parametrize("explicit", [False, True])
def test_recomputation(inputType, explicit):
    accumulationFactor = 3

    data_type, f = (
        (inputType.np_type, 1) if inputType is not None else (np.float32, 0.1)
    )
    dummy_data = np.zeros((2, 2)).astype(data_type)
    data = (
        np.array([i for i in range(accumulationFactor * 2 * 2)]).astype(data_type) * f
    )
    data = np.reshape(data, (accumulationFactor, 2, 2))

    weight_data = np.array([i for i in range(2 * 2)]).astype(np.float32) * 0.25
    weight_data = np.reshape(weight_data, (2, 2))

    def run_test(enable_recomputation):
        weights = {}

        def init_builder(builder):
            d0 = builder.addInputTensor(dummy_data, "data0")
            w0 = builder.addInitializedInputTensor(weight_data)
            weights[w0] = np.empty(shape=weight_data.shape, dtype=weight_data.dtype)

            if inputType is not None:
                d0_float = builder.aiOnnx.cast([d0], "FLOAT")
                t0 = builder.aiOnnx.mul([d0_float, w0])
            else:
                t0 = builder.aiOnnx.mul([d0, w0])
            t1 = builder.aiOnnx.sigmoid([t0])
            t2 = builder.aiGraphcore.scale([t1], 2.0)
            loss = builder.aiGraphcore.identityloss([t2])

            if inputType is not None:
                builder.virtualGraph(d0_float, 0)

            for t in (t0, t1, t2):
                builder.virtualGraph(t, 0)

            builder.virtualGraph(loss, 1)

            return [loss]

        session = PopartTestSession()
        session.mode = "train"
        session.options.virtualGraphMode = popart.VirtualGraphMode.Manual
        session.options.enablePipelining = True
        if enable_recomputation:
            session.options.autoRecomputation = popart.RecomputationType.Standard
        session.options.accumulationFactor = accumulationFactor
        session.options.enableGradientAccumulation = True
        session.options.enableExplicitIR(explicit)

        with tu.create_test_device(numIpus=2) as device:
            session.prepare(init_builder, device=device)

            _ = session.run({"data0": data})

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


# Test that pipeline IpuCopyOpx handles internal aliases correctly. Expectation
# that the ConatOp output contains such internal aliases and the pipelined
# program compiles successfully.
@pytest.mark.parametrize("explicit", [False, True])
def test_internal_alias_ipucopy(explicit):
    builder = popart.Builder()

    with builder.virtualGraph(0), builder.pipelineStage(0):
        model_input = builder.addInputTensor(popart.TensorInfo("FLOAT", [1, 2, 1]))
        concat = builder.aiOnnx.concat([model_input, model_input], axis=1)

    with builder.virtualGraph(1), builder.pipelineStage(1):
        result = builder.aiOnnx.add([concat, concat])

    opts = popart.SessionOptions()
    opts.enablePipelining = True
    opts.virtualGraphMode = popart.VirtualGraphMode.Manual
    opts.enableExplicitIR(explicit)

    with tu.create_test_device(numIpus=2) as device:
        session = popart.InferenceSession(
            fnModel=builder.getModelProto(),
            dataFlow=popart.DataFlow(2, {result: popart.AnchorReturnType("All")}),
            deviceInfo=device,
            userOptions=opts,
        )

        session.prepareDevice()

        feed_dict = {model_input: np.zeros([2, 2, 1], dtype=np.float32)}
        stepio = popart.PyStepIO(feed_dict, session.initAnchorArrays())
        session.run(stepio)


@tu.requires_ipu_model
@pytest.mark.parametrize("explicit", [False, True])
def test_bad_auto_staging(explicit):
    bps = 4
    dummy_data = np.random.rand(2, 2).astype(np.float32)

    def init_builder(builder):
        d0 = builder.addInputTensor(dummy_data, "data0")

        t0 = builder.aiOnnx.sin([d0])
        t1 = builder.aiOnnx.sin([t0])
        t2 = builder.aiOnnx.sin([t1])
        loss = builder.aiGraphcore.identityloss([t2])

        builder.addOutputTensor(loss)

        builder.virtualGraph(t0, 0)
        builder.virtualGraph(t1, 1)
        builder.virtualGraph(t2, 0)
        builder.virtualGraph(loss, 0)

        return [loss]

    session = PopartTestSession()
    session.options.virtualGraphMode = popart.VirtualGraphMode.Manual
    session.options.enablePipelining = True
    session.options.enableExplicitIR(explicit)
    session.device = "ipu_model"
    session.numIPUs = 2
    session.batchesPerStep = bps

    # test a pipeline stage appearing on multiple virtual graphs
    with tu.create_test_device(numIpus=session.numIPUs) as device:
        with pytest.raises(popart.popart_exception) as e_info:
            session.prepare(init_builder, device=device)

    assert e_info.value.args[0].startswith(
        "Tensor Sin:0/1 is consumed in an earlier pipeline stage than it is produced"
    )

    # TODO: T41431
    # The below lines should be uncommented when auto pipeline stage is improved.
    # def ref(d0):
    #     t0 = np.sin(d0)
    #     t1 = np.sin(t0)
    #     t2 = np.sin(t1)
    #     return t2
    # assert len(sessionAnchors) == 1
    # result = [v for k, v in sessionAnchors.items()][0]

    # for i in range(bps):
    #     refResult = ref(data[i])
    #     print(f'Batch {i}: {result[i]}')
    #     print(f'Ref result: {refResult}')
    #     print()

    #     assert np.allclose(result[i], refResult)


@tu.requires_ipu_model
@pytest.mark.parametrize("mode", ["nopipeline", "explicit", "accumulate"])
def test_pipeline_fwd_only_program_errors(mode):
    """
    In this test we check that an error is thrown when trying to use
    createImplicitPipeliningFwdOnlyProgram with explicit pipelining or
    without gradient accumulation
    """
    builder = popart.Builder()
    shape_d = [10]
    d0 = builder.addInputTensor(popart.TensorInfo("FLOAT", shape_d))
    d1 = builder.addInputTensor(popart.TensorInfo("FLOAT", shape_d))
    op0_out = builder.aiOnnx.sin([d0], "s0")
    op1_out = builder.aiOnnx.exp([d1], "r0")
    op2_out = builder.aiOnnx.mul([op0_out, op1_out], "m0")
    builder.addOutputTensor(op2_out)
    opts = popart.SessionOptions()
    opts.enablePipelining = mode != "nopipeline"
    opts.virtualGraphMode = popart.VirtualGraphMode.Manual

    # Setting under test
    opts.createImplicitPipeliningFwdOnlyProgram = True

    # Trigger error with combination of the next two options
    if mode == "explicit":
        opts.enableExplicitIR(True)

    if mode != "accumulate":
        opts.enableGradientAccumulation = True
        opts.accumulationFactor = 5

    builder.pipelineStage(op0_out, 0)
    builder.virtualGraph(op0_out, 0)
    builder.pipelineStage(op1_out, 1)
    builder.virtualGraph(op1_out, 1)
    builder.pipelineStage(op2_out, 2)
    builder.virtualGraph(op2_out, 2)

    with tu.create_test_device(numIpus=3) as device:
        with pytest.raises(popart.popart_exception) as e_info:
            session = popart.TrainingSession(
                fnModel=builder.getModelProto(),
                dataFlow=popart.DataFlow(10, [op2_out, "loss"]),
                loss=op2_out,
                optimizer=popart.ConstSGD(1),
                userOptions=opts,
                deviceInfo=device,
            )
            session.prepareDevice()

        if mode == "nopipeline":
            assert e_info.value.args[0].startswith(
                "Implicit pipelining forward-only program is not supported "
                "without implicit pipelining."
            )
        if mode == "explicit":
            assert e_info.value.args[0].startswith(
                "Implicit pipelining forward-only program is not supported "
                "with explicit pipelining."
            )
        if mode == "accumulate":
            assert e_info.value.args[0].startswith(
                "Implicit pipelining forward-only program is not supported "
                "without gradient accumulation."
            )
