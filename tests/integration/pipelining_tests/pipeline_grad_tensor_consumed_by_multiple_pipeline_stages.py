# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart
import numpy as np

# `import test_util` requires adding to sys.path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu


@tu.requires_ipu_model
def test_grad_tensor_consumed_by_multiple_pipeline_stages_recompute():
    """
    The model:

         VG0, PS0       VG1, PS1      VG0, PS2      VG0, PS2
    w0 ---.-----------------------------.
          Matmul -- t0 -- Sqrt -- t1 -- Add -- t2 -- NllLoss -- loss
    in0 --'                                         /
    in1 -------------------------------------------

    Importantly, the weight, w0, is shared by the two ops on the same
    VirtualGraph, but with different PipelineStages. This means in the
    backwards pass, you get the structure:

        VG0, PS2
    t2 --.                      VG0, PS2
         NllGrad -- grad_t2 -- IdentityCopy
    in1 -'            \    VG0, PS4
                        --- GradSum -- grad_w0
                 grad_t0 -----'

    This means, either:
    (1) grad_t2 must be stashed in PipelineStage2, and restored in
        PipelineStage4 before GradSum is computed
    (2) in1 and t2 need to be stashed in PipelineStage2 and restored in
        PipelineStage4, and NllGrad recomputed before GradSum is computed.
    We do (1) regardless of the value of 'opts.autoRecomputation'
    """
    bps = 5
    dshape = [1, 4, 4]
    lshape = [1, 4]
    in0_data = np.random.rand(bps, *dshape).astype(np.float32)
    in1_data = np.random.randint(5, size=[5, 4])
    w0_data = np.random.rand(*dshape).astype(np.float32)

    pipeline = True
    recompute = True

    def runModel(pipeline, recompute):
        builder = popart.Builder()
        in0 = builder.addInputTensor("FLOAT", dshape)
        in1 = builder.addInputTensor("INT32", lshape)
        w0 = builder.addInitializedInputTensor(w0_data)
        with builder.virtualGraph(0), builder.pipelineStage(0):
            x = builder.aiOnnx.matmul([in0, w0])
        with builder.virtualGraph(1), builder.pipelineStage(1):
            x = builder.aiOnnx.sqrt([x])
        with builder.virtualGraph(0), builder.pipelineStage(2):
            x = builder.aiOnnx.add([w0, x])
            loss = builder.aiGraphcore.nllloss([x, in1])

        opts = popart.SessionOptions()
        opts.virtualGraphMode = popart.VirtualGraphMode.Manual
        opts.enablePipelining = pipeline
        if pipeline == True:
            opts.enableGradientAccumulation = True
            opts.accumulationFactor = bps
            test_bps = 1
        else:
            test_bps = bps

        if recompute == True:
            opts.autoRecomputation = popart.RecomputationType.Pipeline

        session = popart.TrainingSession(
            deviceInfo=popart.DeviceManager().createIpuModelDevice(
                {"numIPUs": "2"}),
            dataFlow=popart.DataFlow(test_bps, [loss]),
            fnModel=builder.getModelProto(),
            loss=loss,
            optimizer=popart.ConstSGD(0.1),
            userOptions=opts)

        session.prepareDevice()
        session.weightsFromHost()
        anchors = session.initAnchorArrays()
        stepio = popart.PyStepIO({in0: in0_data, in1: in1_data}, anchors)
        session.run(stepio)

        weights = {}
        weights[w0] = np.empty(shape=dshape, dtype=np.float32)
        weightsIo = popart.PyWeightsIO(weights)
        session.weightsToHost()
        session.readWeights(weightsIo)
        return weights[w0]

    final_weights_ref = runModel(pipeline=True, recompute=False)
    final_weights_pipelinerecomp = runModel(pipeline=True, recompute=True)

    print(w0_data)
    print(final_weights_ref)
    print(final_weights_pipelinerecomp)
    assert np.allclose(final_weights_ref, final_weights_pipelinerecomp)
