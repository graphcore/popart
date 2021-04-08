import popart
import numpy as np
import pytest

# `import test_util` requires adding to sys.path
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu


def test_implicit_recompute_op_scheduled_pre_loss_no():
    """
    Regression test for T36828. Confirm that compilation completes without an
    exception being thrown.

    It is possible that the MulGrad op that produces Gradient___t3 is scheduled
    early (e.g. at index 0 in the schedule). If this happens, all ops after it
    in the schedule are classified as 'post loss'.

    The matmul operation is recomputed in the backwards pass. The implicit
    recomputation setting forbids that an op to be recomputed is a 'post loss'
    op.
    """
    builder = popart.Builder()
    t0 = builder.addInputTensor("FLOAT", [2, 2])
    t1 = builder.addInitializedInputTensor(
        np.random.rand(2, 2).astype(np.float32))
    t2 = builder.aiOnnx.matmul([t0, t1])
    t3 = builder.aiGraphcore.l1loss([t2], 0.1)

    const = np.array([4]).astype(np.float32)
    t5 = builder.aiOnnx.constant(const)
    t6 = builder.aiOnnx.mul([t3, t5])

    builder.recomputeOutputInBackwardPass(t2)

    session = popart.TrainingSession(deviceInfo=tu.create_test_device(),
                                     fnModel=builder.getModelProto(),
                                     dataFlow=popart.DataFlow(1, []),
                                     loss=t6,
                                     optimizer=popart.SGD(
                                         {"lossScaling": (2.0, False)}))

    session.prepareDevice()
