# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import numpy as np
import pytest
import popart
import pprint
import json

# `import test_util` requires adding to sys.path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu


def test_sgd_mixed_mode(tmpdir):

    w0name = "__w0__"
    w1name = "__w1__"
    w2name = "__w2__"

    sampleDim = 1
    batchSize = 1
    stepSize = 1

    def test(opt0, opt1, e0, e1, e2):
        builder = popart.Builder()

        input0Shape = [stepSize, batchSize, sampleDim]
        input0 = builder.addInputTensor(
            popart.TensorInfo("FLOAT", input0Shape), "input0")

        w0data = np.array([100.0], dtype=np.float32)
        w0R = np.array([-777.0], dtype=np.float32)
        w0Id = builder.addInitializedInputTensor(w0data, w0name)

        w1data = np.array([200.0], dtype=np.float32)
        w1R = np.array([-777.0], dtype=np.float32)
        w1Id = builder.addInitializedInputTensor(w1data, w1name)

        w2data = np.array([300.0], dtype=np.float32)
        w2R = np.array([-777.0], dtype=np.float32)
        w2Id = builder.addInitializedInputTensor(w2data, w2name)

        add0 = builder.aiOnnx.add([w0Id, input0])
        add1 = builder.aiOnnx.add([w1Id, add0])
        add2 = builder.aiOnnx.add([w2Id, add1])

        builder.addOutputTensor(add2)

        proto = builder.getModelProto()

        dataFlow = popart.DataFlow(1, {})

        opts = popart.SessionOptions()
        opts.reportOptions = {"showExecutionSteps": "true"}
        opts.enableGroupedMatmuls = False

        pat = popart.Patterns(popart.PatternsLevel.DEFAULT)

        session = popart.TrainingSession(
            fnModel=proto,
            dataFeed=dataFlow,\
            userOptions=opts,
            losses=[popart.L1Loss(add2, "l1LossVal", 1.0)],
            optimizer=opt0,
            passes=pat,
            deviceInfo=tu.create_test_device(opts={"compileIPUCode": False}))

        session.prepareDevice()

        session.weightsFromHost()

        anchors = session.initAnchorArrays()

        input0Data = np.array([3.1415], dtype=np.float32)

        stepio = popart.PyStepIO({input0: input0Data}, anchors)
        session.optimizerFromHost()
        session.run(stepio)

        session.updateOptimizer(opt1)
        session.optimizerFromHost()
        session.run(stepio)

        session.weightsToHost()

        weightsRead = popart.PyWeightsIO({w0Id: w0R, w1Id: w1R, w2Id: w2R})

        session.readWeights(weightsRead)

        assert (np.isclose(e0['initalValue'], w0R))
        assert (np.isclose(e1['initalValue'], w1R))
        assert (np.isclose(e2['initalValue'], w2R))

    # Test 1 (same as C++ test)
    defaultWeightDecay = (0, False)
    defaultLearningRate = (0.1, True)
    lossScaling = (10, True)

    opt0 = popart.SGD({
        "defaultLearningRate": defaultLearningRate,
        "defaultWeightDecay": defaultWeightDecay,
        "lossScaling": lossScaling
    })

    opt0.insertSpecific(w1name, {
        "weightDecay": (0, True),
        "learningRate": (0.2, False)
    })

    opt1 = popart.SGD({
        "defaultLearningRate": defaultLearningRate,
        "defaultWeightDecay": defaultWeightDecay,
        "lossScaling": lossScaling
    })

    opt1.insertSpecific(w1name, {
        "weightDecay": (0, True),
        "learningRate": (0.5, False)
    })

    e0 = {
        'initalValue': 100.0 - 0.1 - 0.1,
        'constSlr': True,
        'constSwdf': False
    }
    e1 = {
        'initalValue': 200.0 - 0.2 - 0.5,
        'constSlr': False,
        'constSwdf': False
    }
    e2 = {
        'initalValue': 300.0 - 0.1 - 0.1,
        'constSlr': True,
        'constSwdf': False
    }

    test(opt0, opt1, e0, e1, e2)
