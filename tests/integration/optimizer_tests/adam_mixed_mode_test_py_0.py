# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
import onnx
from onnx import numpy_helper
from onnx import TensorProto
import json

# `import test_util` requires adding to sys.path
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu


def to_array(weight):
    if weight.data_type == TensorProto.FLOAT16:
        int_data = np.asarray(weight.int32_data, np.int32)
        np_weight = int_data.view(dtype=np.float16).reshape(weight.dims)
    else:
        np_weight = numpy_helper.to_array(weight)
    return np_weight


def run_adam_mixed_mode(steps, opt_dicts, enable_outlining, tmpdir, dtype=np.float32):
    def run(opt_dict, enable_outlining, model_file_name):
        np.random.seed(1878)
        dsize = 10
        builder = popart.Builder()
        ip = builder.addInputTensor(
            popart.TensorInfo(
                "FLOAT" if dtype == np.float32 else "FLOAT16", [dsize, dsize]
            )
        )

        def add_layer(in_id, name):
            w = builder.addInitializedInputTensor(
                np.random.rand(dsize, dsize).astype(dtype), "w_" + name
            )
            b = builder.addInitializedInputTensor(
                np.random.rand(dsize).astype(dtype), "b_" + name
            )
            matmul_id = builder.aiOnnx.gemm([in_id, w, b], 1, 1, False, False)
            return matmul_id

        m1 = add_layer(ip, "0")
        m2 = add_layer(m1, "1")
        m3 = add_layer(m2, "2")
        m4 = add_layer(m3, "3")

        out = builder.aiGraphcore.identityloss([m4])
        builder.addOutputTensor(out)

        with tu.create_test_device() as device:

            anchors = {}

            opts = popart.SessionOptions()
            opts.enableOutliningCopyCostPruning = False
            opts.outlineThreshold = -np.inf
            opts.enableOutlining = enable_outlining

            proto = builder.getModelProto()

            session = popart.TrainingSession(
                fnModel=proto,
                dataFlow=popart.DataFlow(1, anchors),
                optimizer=opt_dict[0],
                loss=out,
                patterns=popart.Patterns(popart.PatternsLevel.All),
                userOptions=opts,
                deviceInfo=device,
            )

            session.prepareDevice()
            session.weightsFromHost()

            for i in range(steps):
                if i in opt_dict:
                    session.updateOptimizerFromHost(opt_dict[i])
                ip_data = np.ones((dsize, dsize), dtype=dtype)
                stepio = popart.PyStepIO({ip: ip_data}, anchors)
                session.run(stepio)

            session.modelToHost(str(tmpdir / model_file_name))

    for i, opt_dict in enumerate(opt_dicts):
        print(f"Running adam_mixed_mode_{i}")
        run(opt_dict, enable_outlining[i], f"adam_mixed_mode_{i}.onnx")

    gt_onnx = onnx.load(str(tmpdir / "adam_mixed_mode_0.onnx"))

    for i, opt_dict in enumerate(opt_dicts):
        print(f"Testing run adam_mixed_mode_{i}")
        val_onnx = onnx.load(str(tmpdir / f"adam_mixed_mode_{i}.onnx"))
        for j in range(len(gt_onnx.graph.initializer)):
            print(f"Checking initializer {j}")
            gt = gt_onnx.graph.initializer[j]
            gt = to_array(gt)
            val = val_onnx.graph.initializer[j]
            val = to_array(val)
            # print(gt, val)
            assert np.allclose(gt, val)


# Test Adam with different parameters constant / non-constant
def test_adam_mixed_mode_0(tmpdir):

    # optimizer parameters
    defaultLearningRate = 0.005
    defaultBeta1 = 0.7
    defaultBeta2 = 0.8
    defaultWeightDecay = 0.1
    defaultEps = 1e-6
    lossScaling = 10.0

    optMaps = [
        {
            0: popart.Adam(
                {
                    "defaultLearningRate": (defaultLearningRate, True),
                    "defaultBeta1": (defaultBeta1, True),
                    "defaultBeta2": (defaultBeta2, True),
                    "defaultWeightDecay": (defaultWeightDecay, True),
                    "defaultEps": (defaultEps, True),
                    "lossScaling": (lossScaling, True),
                }
            )
        }
    ]
    outlining = [False]

    for i in range(6):
        optMap = {
            "defaultLearningRate": (defaultLearningRate, i != 0),
            "defaultBeta1": (defaultBeta1, i != 1),
            "defaultBeta2": (defaultBeta2, i != 2),
            "defaultWeightDecay": (defaultWeightDecay, i != 3),
            "defaultEps": (defaultEps, i != 4),
            "lossScaling": (lossScaling, i != 5),
        }
        optMaps = optMaps + [{0: popart.Adam(optMap)}]
        outlining = outlining + [False]

    for i in range(6):
        optMap = {
            "defaultLearningRate": (defaultLearningRate, i != 0),
            "defaultBeta1": (defaultBeta1, i != 1),
            "defaultBeta2": (defaultBeta2, i != 2),
            "defaultWeightDecay": (defaultWeightDecay, i != 3),
            "defaultEps": (defaultEps, i != 4),
            "lossScaling": (lossScaling, i != 5),
        }
        optMaps = optMaps + [{0: popart.Adam(optMap)}]
        outlining = outlining + [True]

    run_adam_mixed_mode(10, optMaps, outlining, tmpdir, np.float32)
    run_adam_mixed_mode(10, optMaps, outlining, tmpdir, np.float16)


# Test Adam with weight specific const and non-const parameters
def test_adam_mixed_mode_1(tmpdir):

    # optimizer parameters
    defaultLearningRate0 = 0.005
    defaultLearningRate5 = 0.0025

    defaultBeta1 = 0.7
    defaultBeta2 = 0.8
    defaultWeightDecay = 0.1
    defaultEps = 1e-6
    lossScaling = 10.0

    adam00 = popart.Adam(
        {
            "defaultLearningRate": (defaultLearningRate0, False),
            "defaultBeta1": (defaultBeta1, True),
            "defaultBeta2": (defaultBeta2, True),
            "defaultWeightDecay": (defaultWeightDecay, True),
            "defaultEps": (defaultEps, True),
            "lossScaling": (lossScaling, True),
        }
    )

    adam00.insertSpecific("w_0", {"beta1": (0.9, True), "beta2": (0.99, True)})
    adam00.insertSpecific("b_0", {"beta1": (0.9, True), "beta2": (0.99, True)})

    adam05 = popart.Adam(
        {
            "defaultLearningRate": (defaultLearningRate5, False),
            "defaultBeta1": (defaultBeta1, True),
            "defaultBeta2": (defaultBeta2, True),
            "defaultWeightDecay": (defaultWeightDecay, True),
            "defaultEps": (defaultEps, True),
            "lossScaling": (lossScaling, True),
        }
    )

    adam05.insertSpecific("w_0", {"beta1": (0.9, True), "beta2": (0.99, True)})
    adam05.insertSpecific("b_0", {"beta1": (0.9, True), "beta2": (0.99, True)})

    adam10 = popart.Adam(
        {
            "defaultLearningRate": (defaultLearningRate0, False),
            "defaultBeta1": (defaultBeta1, False),
            "defaultBeta2": (defaultBeta2, False),
            "defaultWeightDecay": (defaultWeightDecay, False),
            "defaultEps": (defaultEps, False),
            "lossScaling": (lossScaling, False),
        }
    )

    adam10.insertSpecific("w_0", {"beta1": (0.9, False), "beta2": (0.99, False)})
    adam10.insertSpecific("b_0", {"beta1": (0.9, False), "beta2": (0.99, False)})

    adam15 = popart.Adam(
        {
            "defaultLearningRate": (defaultLearningRate5, False),
            "defaultBeta1": (defaultBeta1, False),
            "defaultBeta2": (defaultBeta2, False),
            "defaultWeightDecay": (defaultWeightDecay, False),
            "defaultEps": (defaultEps, False),
            "lossScaling": (lossScaling, False),
        }
    )

    adam15.insertSpecific("w_0", {"beta1": (0.9, False), "beta2": (0.99, False)})
    adam15.insertSpecific("b_0", {"beta1": (0.9, False), "beta2": (0.99, False)})

    adam20 = popart.Adam(
        {
            "defaultLearningRate": (defaultLearningRate0, False),
            "defaultBeta1": (defaultBeta1, True),
            "defaultBeta2": (defaultBeta2, False),
            "defaultWeightDecay": (defaultWeightDecay, False),
            "defaultEps": (defaultEps, False),
            "lossScaling": (lossScaling, False),
        }
    )

    adam20.insertSpecific("w_0", {"beta1": (0.9, False), "beta2": (0.99, True)})
    adam20.insertSpecific("b_0", {"beta1": (0.9, False), "beta2": (0.99, True)})

    adam25 = popart.Adam(
        {
            "defaultLearningRate": (defaultLearningRate5, False),
            "defaultBeta1": (defaultBeta1, True),
            "defaultBeta2": (defaultBeta2, False),
            "defaultWeightDecay": (defaultWeightDecay, False),
            "defaultEps": (defaultEps, False),
            "lossScaling": (lossScaling, False),
        }
    )

    adam25.insertSpecific("w_0", {"beta1": (0.9, False), "beta2": (0.99, True)})
    adam25.insertSpecific("b_0", {"beta1": (0.9, False), "beta2": (0.99, True)})

    # Change Adam optimizer after 0 and 5 steps
    optMaps = [{0: adam00, 5: adam05}, {0: adam10, 5: adam15}, {0: adam20, 5: adam25}]

    outlining = [True, True, True]

    run_adam_mixed_mode(10, optMaps, outlining, tmpdir, np.float32)

    debug_filename = str(tmpdir) + "/debug.json"
    popart.initializePoplarDebugInfo(debug_filename, "json")

    run_adam_mixed_mode(10, optMaps, outlining, tmpdir, np.float16)

    # Ensure Adam operations have correct debug context
    popart.closePoplarDebugInfo()
    num_adams = 0
    parents = set()
    with open(debug_filename, encoding="utf-8") as json_file:
        data = json.load(json_file)
        for context in data["contexts"]:
            if (
                context["layer"] == "popart"
                and "opid" in context
                and "Adam" in context["opid"]
            ):
                parents.add(context["parentId"])
                num_adams += 1
        for context in data["contexts"]:
            if context["id"] in parents:
                parents.remove(context["id"])
                assert context["layer"] == "popartbuilder"
                assert data["stringTable"][context["name"]] == "adam"
    assert num_adams == 84
    assert len(parents) == 0
