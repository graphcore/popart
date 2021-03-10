# 'import test_util' requires adding to sys.path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
import popart
import test_util as tu


@tu.requires_ipu
def test_loss_grad_scaling_with_replication():
    """
    Run two models: (1) an L1LossOp with mean reduction and (2) an
    L1 loss op with mean reduction, followed by an IdentityLossOp with mean
    reduction.

    The IdentityLossOp should be a nop - i.e. the input gradient should be
    identitcal between the two models. Observe that this is not the case
    when replication factor > 1, unless the 'accumulationAndReplicationReductionType'
    SessionOption is used
    """
    t0_shape = [4, 4]
    reduction = popart.ReductionType.Mean
    repl_factor = 2
    t0_data = np.random.rand(repl_factor, *t0_shape).astype(np.float32)
    builder = popart.Builder()
    t0 = builder.addInputTensor("FLOAT", t0_shape)
    t1 = builder.aiGraphcore.l1loss([t0], 0.1, reduction=reduction)
    p0 = builder.getModelProto()

    t2 = builder.aiGraphcore.identityloss([t1], reduction=reduction)
    p1 = builder.getModelProto()

    def run_session(proto, loss_id, legacyOptions=True):
        anchor_id = popart.reservedGradientPrefix() + t0

        opts = popart.SessionOptions()
        opts.enableReplicatedGraphs = True
        opts.replicatedGraphCount = repl_factor
        if not legacyOptions:
            opts.accumulationAndReplicationReductionType = reduction

        session = popart.TrainingSession(
            fnModel=proto,
            deviceInfo=tu.create_test_device(repl_factor),
            dataFlow=popart.DataFlow(1, [anchor_id]),
            loss=loss_id,
            optimizer=popart.ConstSGD(0.1),
            userOptions=opts)

        session.prepareDevice()
        anchors = session.initAnchorArrays()
        stepio = popart.PyStepIO({t0: t0_data}, anchors)
        session.run(stepio)
        return anchors[anchor_id]

    def compare_sessions(legacyOptions):
        result0 = run_session(p0, t1, legacyOptions)
        result1 = run_session(p1, t2, legacyOptions)
        return np.allclose(result0, result1)

    # TODO legacy behaviour will be removed after T33184
    assert not compare_sessions(legacyOptions=True)

    assert compare_sessions(legacyOptions=False)


optimizers = []
optimizers.append(popart.ConstSGD(0.1))  # const loss scaling
optimizers.append(popart.SGD({"lossScaling":
                              (2.5, False)}))  # variable loss scaling


@pytest.mark.parametrize("optimizer", optimizers)
@tu.requires_ipu
def test_loss_grad_scaling_with_replication_2(optimizer):
    """
    w2 ----------------.
                        \
    t0 - Matmul - t1 - Matmul - t2 - L1Loss0 - t3 - Add - t6 - IdentityLoss - t7
         |         \                                 |
    w0 --'          `- Matmul - t4 - L1Loss1 - t5 ---'
                        |
    w1 -----------------'

    with gradient accumulation and graph replication

    1. Legacy options
    - Loss0,1 reduction types: mean
    - IdenityLoss reduction type: none
    - accumulationReductionType = ReductionType::Mean
    - accumulationAndReplicationReductionType = ReductionType::NoReduction

    2. New options
    - Loss0,1 reduction types: mean
    - IdenityLoss reduction type: none
    - accumulationReductionType = ReductionType::Mean
    - accumulationAndReplicationReductionType = ReductionType::Mean
    """
    builder = popart.Builder()
    t0_shape = [2, 1, 2, 2]
    w_shape = [1, 1, 2, 2]
    w0_data = np.random.rand(*w_shape).astype(np.float32)
    w1_data = np.random.rand(*w_shape).astype(np.float32)
    w2_data = np.random.rand(*w_shape).astype(np.float32)
    t0 = builder.addInputTensor("FLOAT", t0_shape)
    w0 = builder.addInitializedInputTensor(w0_data)
    w1 = builder.addInitializedInputTensor(w0_data)
    w2 = builder.addInitializedInputTensor(w0_data)
    t1 = builder.aiOnnx.matmul([t0, w0])

    t2 = builder.aiOnnx.matmul([t1, w2])
    t3 = builder.aiGraphcore.l1loss([t2],
                                    0.3,
                                    reduction=popart.ReductionType.Mean)

    t4 = builder.aiOnnx.matmul([t1, w1])
    t5 = builder.aiGraphcore.l1loss([t4],
                                    0.2,
                                    reduction=popart.ReductionType.Mean)
    t6 = builder.aiOnnx.add([t3, t5])
    t7 = builder.aiGraphcore.identityloss(
        [t4], reduction=popart.ReductionType.NoReduction)

    opts = popart.SessionOptions()
    opts.accumulationReductionType = popart.ReductionType.Mean
    opts.enableGradientAccumulation = True
    accl = 3
    opts.accumulationFactor = accl
    opts.enableReplicatedGraphs = True
    repl = 2
    opts.replicatedGraphCount = repl

    bps = 2
    t0_data = np.random.rand(bps, repl, accl, *t0_shape).astype(np.float32)

    def getAnchors(options):
        session = popart.TrainingSession(
            fnModel=builder.getModelProto(),
            dataFlow=popart.DataFlow(
                bps, [t7, popart.reservedGradientPrefix() + t0]),
            optimizer=optimizer,
            loss=t5,
            userOptions=options,
            deviceInfo=tu.create_test_device(repl))
        session.prepareDevice()
        session.weightsFromHost()
        anchors = session.initAnchorArrays()
        stepio = popart.PyStepIO({t0: t0_data}, anchors)
        session.run(stepio)

        return anchors

    legacy_reuslts = getAnchors(opts)

    new_opts = opts
    new_opts.accumulationAndReplicationReductionType = popart.ReductionType.Mean
    new_reuslts = getAnchors(new_opts)

    for key in legacy_reuslts.keys():
        assert np.allclose(legacy_reuslts[key], new_reuslts[key])
