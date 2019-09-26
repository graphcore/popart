import numpy as np
import pytest
import popart
import pprint
import json
import platform

# 'import test_util' requires adding to sys.path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import test_util as tu


def test_full_recompute_pipelining(tmpdir):
    batches_per_step = 5
    batch_size = 6
    hidden_size = 12

    lhs_shape = [batch_size, hidden_size]
    rhs_shape = [hidden_size, hidden_size]
    mask_shape = [hidden_size]

    lhs_data = np.ones((batches_per_step, *lhs_shape), dtype=np.float32)
    rhs_data = np.ones((*rhs_shape, ), dtype=np.float32)

    r = np.arange(0, mask_shape[0])
    masks = []
    for i in range(batches_per_step):
        masks.append(np.less(r, i).astype(np.float32))
    mask_data = np.stack(masks)

    def run_test(verify):
        builder = popart.Builder()

        lhs = builder.addInputTensor(popart.TensorInfo("FLOAT", lhs_shape),
                                     "lhs")
        mask = builder.addInputTensor(popart.TensorInfo("FLOAT", mask_shape),
                                      "mask")
        rhs1 = builder.addInitializedInputTensor(rhs_data, "rhs")
        rhs2 = builder.addInitializedInputTensor(rhs_data, "rhs")

        # Recomp Mode Standard would reject "mask" as an stash op
        with builder.virtualGraph(0), builder.pipelineStage(0):
            o = builder.aiOnnx.add([lhs, mask])
            o = builder.aiOnnx.matmul([o, rhs1])
        with builder.virtualGraph(1), builder.pipelineStage(1):
            o = builder.aiOnnx.matmul([o, rhs2])
        with builder.virtualGraph(2), builder.pipelineStage(2):
            o = builder.aiOnnx.add([o, mask])

        loss = popart.L1Loss(o, "l1LossVal", 0.1)
        loss.virtualGraph(2)
        loss.pipelineStage(2)

        proto = builder.getModelProto()

        dataFlow = popart.DataFlow(batches_per_step,
                                   {o: popart.AnchorReturnType("ALL")})

        opts = popart.SessionOptions()
        opts.enableOutlining = False
        opts.enablePipelining = True
        opts.autoRecomputation = popart.RecomputationType.Pipeline
        opts.virtualGraphMode = popart.VirtualGraphMode.Manual

        pat = popart.Patterns(popart.PatternsLevel.DEFAULT)

        session = popart.TrainingSession(fnModel=proto,
                                         dataFeed=dataFlow,
                                         userOptions=opts,
                                         losses=[loss],
                                         optimizer=popart.ConstSGD(0.01),
                                         passes=pat,
                                         deviceInfo=tu.get_ipu_model(
                                             compileIPUCode=False, numIPUs=3))

        session.prepareDevice()

        session.weightsFromHost()
        session.optimizerFromHost()

        anchors = session.initAnchorArrays()

        inputs = {lhs: lhs_data, mask: mask_data}
        stepio = popart.PyStepIO(inputs, anchors)

        session.run(stepio)
        session.weightsToHost()

        verify(session)

    def verify(session):
        ''' Verify the the matmul in the main graphs is correct'''
        ir = json.loads(session._serializeIr(
            popart.IrSerializationFormat.JSON))
        stashes = [op for op in ir["maingraph"] if op["type"] == "Stash"]
        stashedTensors = [stash["inputs"][0]["name"] for stash in stashes]

        assert ('lhs' in stashedTensors)
        assert ('mask' in stashedTensors)

    run_test(verify)
