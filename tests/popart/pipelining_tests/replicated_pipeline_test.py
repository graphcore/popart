import numpy as np
import pytest
import popart

# `import test_util` requires adding to sys.path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu
from operators_test.op_tester import op_tester

REPL_FACTOR = 2
BPS = 8


def get_model_anchors(doSharding,
                      doPipelining,
                      batchesPerStep,
                      doTraining,
                      replicated_graph_count=1,
                      doProfiling=False,
                      doDropout=False,
                      doGradientAccl=False,
                      acclSteps=1,
                      doDevicex=True,
                      anchorRestoredTensors=False,
                      returnRawInput=False):
    np.random.seed(seed=1)

    builder = popart.Builder()
    batchSize = 16
    microBatchSize = batchSize // acclSteps

    shape_d0 = [microBatchSize, 2, 4, 4]
    shape_l0 = [microBatchSize]

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
    r0 = builder.reshape_const(builder.aiOnnx, [c0], [microBatchSize, 32])
    if doDropout:
        do0 = builder.aiOnnx.dropout([r0], num_outputs=1, ratio=0.2)[0]
        out = builder.aiOnnx.softmax([do0], axis=1, debugPrefix="sfm")
    else:
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

    opts = popart.SessionOptions()
    opts.reportOptions = {"showExecutionSteps": "true"}
    opts.enablePipelining = doPipelining
    opts.enableGradientAccumulation = doGradientAccl
    opts.accumulationFactor = acclSteps
    opts.enableStochasticRounding = False

    if doSharding is False:
        numIPUs = 1 * replicated_graph_count
    else:
        opts.enableVirtualGraphs = True
        numIPUs = 2 * replicated_graph_count
        builder.virtualGraph(s0, 0)
        builder.virtualGraph(e0, 0)
        builder.virtualGraph(c0, 0)
        builder.virtualGraph(r0, 1)
        if doDropout:
            builder.virtualGraph(do0, 1)
        builder.virtualGraph(out, 1)

        loss.virtualGraph(1)
    if replicated_graph_count > 1:
        opts.replicatedGraphCount = replicated_graph_count
        opts.enableReplicatedGraphs = True

    if tu.ipu_available(numIPUs):
        device = tu.acquire_ipu(numIPUs=numIPUs)
    else:
        pytest.skip("Test needs to run on IPU, but none are available")

    if doTraining is True:
        session = popart.TrainingSession(fnModel=builder.getModelProto(),
                                         dataFeed=popart.DataFlow(
                                             batchesPerStep, anchor_map),
                                         losses=[loss],
                                         optimizer=popart.ConstSGD(0.01),
                                         userOptions=opts,
                                         deviceInfo=device)
    else:
        session = popart.InferenceSession(fnModel=builder.getModelProto(),
                                          dataFeed=popart.DataFlow(
                                              batchesPerStep, anchor_map),
                                          losses=[loss],
                                          userOptions=opts,
                                          deviceInfo=device)

    if doDevicex is False:
        return None

    session.prepareDevice()
    anchors = session.initAnchorArrays()
    session.setRandomSeed(0)

    classes = np.prod(shape_d0) // (batchSize * batchesPerStep)

    label = np.random.randint(low=0, high=classes,
                              size=shape_l0).astype(np.int32)

    # With all options enabled return anchors are of the shape:
    # [batches_per_step, accl_factor, repl_factor, micro_batch, *data_shape]
    if acclSteps > 1:
        shape_d0.insert(0, acclSteps)
        label = label.reshape([acclSteps, -1])
    if batchesPerStep > 1:
        shape_d0.insert(0, batchesPerStep)
        label = np.repeat(label[np.newaxis], batchesPerStep, 0)

    data = np.random.random_sample(shape_d0).astype(np.float32)

    # This is a slightly odd case - we want the same data to be input for both
    # replicated graphs, but the dimension we need to repeat on is either the
    # first or second (the replication dimension) depending on whether we
    # have gradient accumulation enabled.
    # If we are not testing, this is a lot simpler as we can split samples however
    # we want.
    if replicated_graph_count > 1:
        if acclSteps > 1:
            data = np.repeat(data[np.newaxis], replicated_graph_count, 2)
            label = label.reshape([replicated_graph_count, -1])
        else:
            data = np.repeat(data[np.newaxis], replicated_graph_count, 1)
            label = label.reshape([replicated_graph_count, -1])

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


def compare_anchors_repl(no_repl_anchors, repl_anchors):
    """
    Compare the tensors in arg1 vs arg2. This is essentially just testing
    tensor vs tensor; batch vs batch, but done once for each replication
    against the 'master' without replication. The first run has replication
    disabled, the second replication enabled.
    """
    # Tensor loop:
    for (tId1, t1), (tId2, t2) in zip(no_repl_anchors.items(),
                                      repl_anchors.items()):
        # Batch loop:
        for i in range(BPS):
            # Replication loop:
            print(f"no replication, batch: {i} {tId1} {np.sum(t1[i])}")
            for j in range(REPL_FACTOR):
                print(
                    f"\treplication {j},  batch: {i} {tId2} {np.sum(t2[i, j])}"
                )
                # Test the relevant batch and replication against the run with
                # no replication.
                assert np.allclose(t1[i], t2[i, j]), \
                    f"""Arrays {tId1} replication {j} batch {i} not equal.
                Max difference {np.max(np.abs(t1[i] - t2[i, j]))}"""


def compare_anchors_both_repl(no_pipe_anchors, pipe_anchors):
    """
    Compare the tensors in arg1 vs arg2. This is essentially just testing
    tensor vs tensor; batch vs batch, but done once for each replication
    against the non pipelined equivalent replication. Both runs must
    have replication enabled, but not gradient accumulation.
    """
    # Tensor loop:
    for (tId1, t1), (tId2, t2) in zip(no_pipe_anchors.items(),
                                      pipe_anchors.items()):
        # Batch loop:
        for i in range(BPS):
            # Replication loop:
            for j in range(REPL_FACTOR):
                print(f"no pipelining replication {j}, batch: {i} \
                    {tId1} {np.sum(t1[i, j])}")
                print(f"pipelining replication    {j}, batch: {i} \
                    {tId2} {np.sum(t2[i, j])}")
                # Test each equivalent replication against each equivalent batch
                assert np.allclose(t1[i, j], t2[i, j]), \
                    f"""Arrays {tId1} replication {j} batch {i} not equal.
                Max difference {np.max(np.abs(t1[i, j] - t2[i, j]))}"""


def compare_anchors_pipe(no_pipe_anchors, pipe_anchors):
    """
    Compare the tensors in arg1 vs arg2. This is essentially just testing
    tensor vs tensor; batch vs batch, but done once for each replication
    against the non pipelined equivalent replication. Both runs must
    have replication and gradient accumulation enabled.
    """
    # Tensor loop:
    for (tId1, t1), (tId2, t2) in zip(no_pipe_anchors.items(),
                                      pipe_anchors.items()):
        # Batch loop:
        for i in range(BPS):
            # Replication loop:
            for j in range(REPL_FACTOR):
                print(f"no pipelining replication {j}, batch: {i} \
                    {tId1} {np.sum(t1[i, :, j])}")
                print(f"pipelining replication    {j}, batch: {i} \
                    {tId2} {np.sum(t2[i, :, j])}")
                # Test each equivalent replication against each equivalent batch
                assert np.allclose(t1[i, :, j], t2[i, :, j]), \
                    f"""Arrays {tId1} replication {j} batch {i} not equal.
                Max difference {np.max(np.abs(t1[i, :, j] - t2[i, :, j]))}"""


# TODO see T16010
# @tu.requires_ipu()
@pytest.mark.skip("Test currently failing on hardware")
def test_output_matches_replication_infer():
    """
    Pipelining + No Replication
    vs
    Pipelining + Replication

    Inference only
    """

    no_repl_anchors = get_model_anchors(
        doSharding=True,
        doPipelining=True,
        batchesPerStep=BPS,
        doTraining=False,
        replicated_graph_count=1,
    )
    repl_anchors = get_model_anchors(
        doSharding=True,
        doPipelining=True,
        batchesPerStep=BPS,
        doTraining=False,
        replicated_graph_count=REPL_FACTOR,
    )

    compare_anchors_repl(no_repl_anchors, repl_anchors)


# TODO see T16010
# @tu.requires_ipu()
@pytest.mark.skip("Test currently failing on hardware")
def test_output_matches_pipeline_infer():
    """
    Pipelining + Replication
    vs
    No Pipelining + Replication

    Inference only
    """

    pipe_anchors = get_model_anchors(
        doSharding=True,
        doPipelining=True,
        batchesPerStep=BPS,
        doTraining=False,
        replicated_graph_count=REPL_FACTOR,
    )
    no_pipe_anchors = get_model_anchors(
        doSharding=True,
        doPipelining=False,
        batchesPerStep=BPS,
        doTraining=False,
        replicated_graph_count=REPL_FACTOR,
    )

    compare_anchors_both_repl(pipe_anchors, no_pipe_anchors)


# TODO see T16010
# @tu.requires_ipu()
@pytest.mark.skip("Test currently failing on hardware")
def test_output_matches_pipeline_train():
    """
    No Pipelining + Replication + Gradient Accumulation
    vs
    Pipelining + Replication + Gradient Accumulation

    Training
    """
    no_repl_anchors = get_model_anchors(
        doSharding=True,
        doPipelining=False,
        batchesPerStep=BPS,
        doTraining=True,
        doGradientAccl=True,
        acclSteps=4,
        replicated_graph_count=REPL_FACTOR,
    )
    repl_anchors = get_model_anchors(
        doSharding=True,
        doPipelining=True,
        batchesPerStep=BPS,
        doTraining=True,
        doGradientAccl=True,
        acclSteps=4,
        replicated_graph_count=REPL_FACTOR,
    )

    compare_anchors_pipe(no_repl_anchors, repl_anchors)


"""
The following cases we expect to be non-equal:
    Pipelining + No Replication
    vs
    Pipelining + Replication

    No Pipelining + Replication
    vs
    Pipelining + Replication

    Pipelining + No Replication + Dropout
    vs
    Pipelining + Replication + Dropout

If you pass four data samples through your model in non-replicated mode, you get
 four weight updates, w0 -> w1 -> w2 -> w3
If you pass four data samples through your model with replication factor = 4 you
 get a single weight update: w0 -> w4.
where w4 != w3.  

For the test:

No Pipelining + Replication
vs
Pipelining + Replication

You'd expect this to be non-equal because of the 'continuous updates' in 
 pipelining,
"""
