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
