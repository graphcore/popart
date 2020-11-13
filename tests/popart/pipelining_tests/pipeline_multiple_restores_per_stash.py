import popart
import pytest
import numpy as np

# `import test_util` requires adding to sys.path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu


@tu.requires_ipu_model
def test_2_restores_for_1_stash_not_supported_with_recomp():
    """
    The model:
         IPU0, PS0   IPU0, PS0
                 w0 ---.            IPU1, PS1      IPU0, PS2     IPU0, PS2
    in0 -- Nop - x0 - Matmul --- x0 -- Sqrt --- x1 -- Add -- x2 -- IdLoss -- out
                  \                                   /
                    ---------------------------------
                    \
                     Transpose (for MatmulGrad) -- ...
                            IPU0, PS4

    Important features:
    - x0 is consumed by three ops on the same VirtualGraph, but all on
      different PipelineStages.
    - When in popart.RecomputationType.Pipeline mode, in0 will be stashed,
      and requires 2 Restore ops that are run in x0's recomputation fragments
    """

    bps = 5
    dshape = [1, 4, 4]
    w0_data = np.random.rand(*dshape).astype(np.float32)

    builder = popart.Builder()
    in0 = builder.addInputTensor("FLOAT", dshape)
    w0 = builder.addInitializedInputTensor(w0_data)
    with builder.virtualGraph(0), builder.pipelineStage(0):
        in0 = builder.aiGraphcore.nop([in0])
        x = builder.aiOnnx.matmul([in0, w0])
    with builder.virtualGraph(1), builder.pipelineStage(1):
        x = builder.aiOnnx.sqrt([x])
    with builder.virtualGraph(0), builder.pipelineStage(2):
        x = builder.aiOnnx.add([in0, x])
        loss = builder.aiGraphcore.identityloss([x])

    opts = popart.SessionOptions()
    opts.virtualGraphMode = popart.VirtualGraphMode.Manual
    opts.enablePipelining = True
    opts.autoRecomputation = popart.RecomputationType.Pipeline

    with pytest.raises(popart.popart_internal_exception) as e_info:
        session = popart.TrainingSession(
            deviceInfo=tu.create_test_device(numIpus=2),
            dataFlow=popart.DataFlow(bps, [loss, w0],
                                     popart.AnchorReturnType("Final")),
            fnModel=builder.getModelProto(),
            loss=loss,
            optimizer=popart.ConstSGD(0.1),
            userOptions=opts)
    assert e_info.value.args[
        0] == "To-stash tensor input requires >1 Restore op. This is not currently supported."
