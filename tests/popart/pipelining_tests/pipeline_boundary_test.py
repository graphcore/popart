# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import numpy as np
import popart

# importing test_session and test_util requires adding to sys.path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from test_session import PopartTestSession
import test_util as tu


@tu.requires_ipu_model
def test_pipeline_boundary():
    np.random.seed(0)
    input_ = np.random.random_sample([4, 4, 2]).astype('float32')
    w0_init = np.random.random_sample([2, 4]).astype('float32')

    def run(inplaceReshape=True):

        builder = popart.Builder()

        in0 = builder.addInputTensor(popart.TensorInfo("FLOAT", [4, 2]))
        w0 = builder.addInitializedInputTensor(w0_init)

        actIn = in0

        with builder.virtualGraph(0):
            actIn = builder.aiOnnx.matmul([actIn, w0])
            actIn = builder.aiOnnx.relu([actIn])
            reshape = builder.aiOnnx.constant(np.asarray([2, 8]))
            actIn = builder.aiOnnx.reshape([actIn, reshape])
            r0 = actIn

        with builder.virtualGraph(1):
            actIn = builder.aiOnnx.identity([actIn], "pipelined_out")

        builder.addOutputTensor(actIn)
        builder.setInplacePreferences(
            r0, {"ReshapeInplace": 10 if inplaceReshape else -10})
        opts = popart.SessionOptions()

        loss = popart.L1Loss(actIn, actIn + "/loss", 0.1)
        loss.virtualGraph(1)
        opts.virtualGraphMode = popart.VirtualGraphMode.Manual
        opts.enablePipelining = True
        numIpus = 2

        patterns = popart.Patterns(popart.PatternsLevel.Default)
        patterns.InPlace = True

        device = tu.create_test_device(numIpus=numIpus)
        session = popart.TrainingSession(fnModel=builder.getModelProto(),
                                         dataFeed=popart.DataFlow(4, [actIn]),
                                         deviceInfo=device,
                                         optimizer=popart.ConstSGD(0.1),
                                         losses=[loss],
                                         userOptions=opts,
                                         patterns=patterns)

        anchors = session.initAnchorArrays()

        inputs = {in0: input_}
        stepio = popart.PyStepIO(inputs, anchors)

        session.prepareDevice()
        session.weightsFromHost()

        session.run(stepio)

        return np.copy(anchors[actIn])

    anchor_pl = run(inplaceReshape=True)
    anchor_no_pl = run(inplaceReshape=False)

    print("Shape: ", anchor_pl.shape)
    for i in range(4):
        assert np.allclose(anchor_pl[i, :], anchor_no_pl[i, :])
        print("Abs diff: ",
              np.max(np.abs(anchor_pl[i, :] - anchor_no_pl[i, :])))
        print("Sum diff: ",
              np.sum(anchor_pl[i, :]) - np.sum(anchor_no_pl[i, :]))

    print("Total Abs diff: ", np.max(np.abs(anchor_pl - anchor_no_pl)))
