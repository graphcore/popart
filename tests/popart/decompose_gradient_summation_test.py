# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import popart
import numpy as np

# importing test_session and test_util requires adding to sys.path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from test_session import PopartTestSession
import test_util as tu


# Model: N matmuls in series, sharing weights
#
# in0 -
#       \
#        Matmul0 - Relu0 - Matmul1 - Relu1 -  ... - MatmulN-1 - ReluN-1 - out
#       /                   /                  /        /
#  w0 -------------------------------------------------
#
# Test:
# Since the weights are shared between N matmuls, the gradient for the
# weight will be a sum of N partial gradients.
# We test here that the optimization to decompose the grad Sum into
# N-1 partial Adds produces the numerically identical weight update
@tu.requires_ipu_model
def test_decompose_gradient_sum():
    def run(doSharding):
        np.random.seed(0)
        builder = popart.Builder()
        shape = [4, 4]
        numLayers = 4
        in0 = builder.addInputTensor(popart.TensorInfo("FLOAT", shape))
        w0_init = np.random.rand(*shape).astype('float32')
        w0 = builder.addInitializedInputTensor(w0_init)

        actIn = in0
        for layer in range(numLayers):
            with builder.virtualGraph(layer if doSharding else 0):
                actIn = builder.aiOnnx.matmul([actIn, w0])
                actIn = builder.aiOnnx.relu([actIn])

        def getUpdatedWeights(decomposeGradSum):
            opts = popart.SessionOptions()
            opts.decomposeGradSum = decomposeGradSum

            loss = popart.L1Loss(actIn, actIn + "/loss", 0.1)
            if doSharding == True:
                loss.virtualGraph(numLayers - 1)
                opts.virtualGraphMode = popart.VirtualGraphMode.Manual
                numIpus = numLayers
            else:
                numIpus = 1

            device = tu.create_test_device(numIpus=numIpus)
            session = popart.TrainingSession(fnModel=builder.getModelProto(),
                                             dataFeed=popart.DataFlow(1, [w0]),
                                             deviceInfo=device,
                                             optimizer=popart.ConstSGD(0.1),
                                             losses=[loss],
                                             userOptions=opts)

            anchors = session.initAnchorArrays()
            np.random.seed(1)  # ensure same input vals between sessions
            inputs = {in0: np.random.rand(*shape).astype('float32')}
            stepio = popart.PyStepIO(inputs, anchors)

            session.prepareDevice()
            session.weightsFromHost()

            session.run(stepio)
            return anchors[w0]

        w1_decomp = getUpdatedWeights(decomposeGradSum=True)
        w1_nodecomp = getUpdatedWeights(decomposeGradSum=False)

        # Sanity check that the weights have been updated
        assert np.allclose(w0_init, w1_decomp) == False

        # Check that the option does not affect the weight update
        assert np.allclose(w1_nodecomp, w1_decomp)

    run(doSharding=False)
    run(doSharding=True)
