# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart
import pytest
import numpy as np

# `import test_util` requires adding to sys.path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu


@tu.requires_ipu_model
@pytest.mark.parametrize("explicit", [False, True])
def test_2_restores_for_1_stash(explicit):
    r"""
    The model:
           IPU0, PS0
     w0 ---.              IPU1, PS1      IPU0, PS2     IPU0, PS2
    in0 --- Matmul --- x0 -- Sqrt --- x1 -- Add -- x2 -- IdLoss -- out
       \                                    /
         ----------------------------------
         \
           Transpose (for MatmulGrad) -- ...
                 IPU0, PS4

    Important feature: in0 is consumed by three ops on the same
    VirtualGraph, but all on different PipelineStages. The pipeline
    transform handles this by adding two Restore ops.
    """

    np.random.seed(2)
    bps = 5
    dshape = [1, 4, 4]
    in0_data = np.random.rand(bps, *dshape).astype(np.float32)
    w0_data = np.random.rand(*dshape).astype(np.float32)

    def runModel(pipeline, recompute):
        builder = popart.Builder()
        in0 = builder.addInputTensor("FLOAT", dshape)
        w0 = builder.addInitializedInputTensor(w0_data)
        with builder.virtualGraph(0), builder.pipelineStage(0):
            x = builder.aiOnnx.matmul([in0, w0])
        with builder.virtualGraph(1), builder.pipelineStage(1):
            x = builder.aiOnnx.sqrt([x])
        with builder.virtualGraph(0), builder.pipelineStage(2):
            x = builder.aiOnnx.add([in0, x])
            loss = builder.aiGraphcore.identityloss([x])

        opts = popart.SessionOptions()
        opts.virtualGraphMode = popart.VirtualGraphMode.Manual
        opts.enablePipelining = pipeline
        opts.enableGradientAccumulation = True
        opts.accumulationFactor = bps
        opts.enableExplicitIR(explicit)
        if recompute == True:
            opts.autoRecomputation = popart.RecomputationType.Pipeline

        with tu.create_test_device(numIpus=2) as device:
            session = popart.TrainingSession(
                deviceInfo=device,
                dataFlow=popart.DataFlow(1, [loss],
                                         popart.AnchorReturnType("Final")),
                fnModel=builder.getModelProto(),
                loss=loss,
                optimizer=popart.ConstSGD(0.1),
                userOptions=opts)

            session.prepareDevice()
            session.weightsFromHost()
            anchors = session.initAnchorArrays()
            stepio = popart.PyStepIO({in0: in0_data}, anchors)
            session.run(stepio)
            session.weightsToHost()
            w0R = np.array(-777.0 * np.ones(w0_data.shape), dtype=np.float32)
            weightsRead = popart.PyWeightsIO({w0: w0R})
            session.readWeights(weightsRead)
            return w0R

    ref_weights = runModel(pipeline=False, recompute=False)
    pipelined_weights = runModel(pipeline=True, recompute=False)
    pipelined_recomp_weights = runModel(pipeline=True, recompute=True)
    print("Untrained        :", w0_data)
    print("Ref              :", ref_weights)
    print("Pipelined        :", pipelined_weights)
    print("Pipelined, recomp:", pipelined_recomp_weights)
    # Verify that the final weights are the same for pipelined vs non-pipelined
    assert np.allclose(ref_weights, pipelined_weights)
    assert np.allclose(ref_weights, pipelined_recomp_weights)


# Test not relevant for explicit pipelining, which should support this through
# multiple recomputation
@tu.requires_ipu_model
def test_2_restores_for_1_stash_not_supported_with_recomp():
    r"""
    The model:
         IPU0, PS0   IPU0, PS0
                 w0 ---.            IPU1, PS1      IPU0, PS2     IPU0, PS2
    in0 -- Nop - x0 - Matmul --- x0 -- Sqrt --- x1 -- Add -- x2 -- IdLoss -- out
                  \                                   /
                    ---------------------------------
                    \
                     Transpose (for MatmulGrad) -- ...
                            IPU0, PS4

    As in the above test except that there is now an Op (Nop) between in0
    and x0 - the tensor that is by Ops on three different PipelineStages.

    When in popart.RecomputationType.Pipeline mode, in0 is still the stashed
    tensor. But in order to compute x0's consumers, we would require two
    recompute fragments:
        - Restore in0 in PS2, recompute x0 for Add's consumption
        - Restore in0 in PS4, recompute x0 Transpose's consumption
    Supporting recomputation of a tensor in >1 fragment is to do in T30014.
    For now we raise an exception
    This should not be an issue with explicit pipelining.
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

    with tu.create_test_device(numIpus=2) as device:
        with pytest.raises(popart.popart_exception) as e_info:
            _ = popart.TrainingSession(deviceInfo=device,
                                       dataFlow=popart.DataFlow(
                                           bps, [loss, w0],
                                           popart.AnchorReturnType("Final")),
                                       fnModel=builder.getModelProto(),
                                       loss=loss,
                                       optimizer=popart.ConstSGD(0.1),
                                       userOptions=opts)
        assert e_info.value.args[0].find(
            "To-stash Tensor '" + in0 +
            "' must be restored for recomputation of a descendent that is not a direct consumer on more than 1 PipelineStage, but this is currently not supported"
        )
