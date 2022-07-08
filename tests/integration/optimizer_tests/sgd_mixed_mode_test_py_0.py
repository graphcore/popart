# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
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
            popart.TensorInfo("FLOAT", input0Shape), "input0"
        )

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

        l1 = builder.aiGraphcore.l1loss([add2], 1.0)

        proto = builder.getModelProto()

        dataFlow = popart.DataFlow(1, {})

        opts = popart.SessionOptions()
        opts.reportOptions = {"showExecutionSteps": "true"}

        pat = popart.Patterns(popart.PatternsLevel.Default)

        with tu.create_test_device(opts={"compileIPUCode": False}) as device:
            session = popart.TrainingSession(
                fnModel=proto,
                dataFlow=dataFlow,
                userOptions=opts,
                loss=l1,
                optimizer=opt0,
                patterns=pat,
                deviceInfo=device,
            )

            session.prepareDevice()

            session.weightsFromHost()

            anchors = session.initAnchorArrays()

            input0Data = np.array([3.1415], dtype=np.float32)

            stepio = popart.PyStepIO({input0: input0Data}, anchors)

            session.run(stepio)

            session.updateOptimizerFromHost(opt1)

            session.run(stepio)

            session.weightsToHost()

            weightsRead = popart.PyWeightsIO({w0Id: w0R, w1Id: w1R, w2Id: w2R})

            session.readWeights(weightsRead)

        assert np.isclose(e0["initalValue"], w0R)
        assert np.isclose(e1["initalValue"], w1R)
        assert np.isclose(e2["initalValue"], w2R)

    # Test 1 (same as C++ test)
    defaultWeightDecay = (0, False)
    defaultLearningRate = (0.1, True)
    lossScaling = (10, True)

    opt0 = popart.SGD(
        {
            "defaultLearningRate": defaultLearningRate,
            "defaultWeightDecay": defaultWeightDecay,
            "lossScaling": lossScaling,
        }
    )

    opt0.insertSpecific(
        w1name, {"weightDecay": (0, True), "learningRate": (0.2, False)}
    )

    opt1 = popart.SGD(
        {
            "defaultLearningRate": defaultLearningRate,
            "defaultWeightDecay": defaultWeightDecay,
            "lossScaling": lossScaling,
        }
    )

    opt1.insertSpecific(
        w1name, {"weightDecay": (0, True), "learningRate": (0.5, False)}
    )

    e0 = {"initalValue": 100.0 - 0.1 - 0.1, "constSlr": True, "constSwdf": False}
    e1 = {"initalValue": 200.0 - 0.2 - 0.5, "constSlr": False, "constSwdf": False}
    e2 = {"initalValue": 300.0 - 0.1 - 0.1, "constSlr": True, "constSwdf": False}

    debug_filename = str(tmpdir) + "/debug.json"
    popart.initializePoplarDebugInfo(debug_filename, "json")

    test(opt0, opt1, e0, e1, e2)

    # Ensure SGD operations have correct debug context
    popart.closePoplarDebugInfo()
    num_sgds = 0
    parents = set()
    with open(debug_filename, encoding="utf-8") as json_file:
        data = json.load(json_file)
        for context in data["contexts"]:
            if (
                context["layer"] == "popart"
                and "opid" in context
                and "SGD" in context["opid"]
            ):
                parents.add(context["parentId"])
                num_sgds += 1
        for context in data["contexts"]:
            if context["id"] in parents:
                parents.remove(context["id"])
                assert context["layer"] == "popart_builder"
                assert data["stringTable"][context["name"]] == "sgd"
    assert num_sgds == 6
    assert len(parents) == 0
