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


def run_adaptive_mixed_mode(
    steps, opt_dicts, enable_outlining, tmpdir, dtype=np.float32
):
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
        print(f"Running adaptive_mixed_mode_{i}")
        run(opt_dict, enable_outlining[i], f"adaptive_mixed_mode_{i}.onnx")

    gt_onnx = onnx.load(str(tmpdir / "adaptive_mixed_mode_0.onnx"))

    for i, opt_dict in enumerate(opt_dicts):
        print(f"Testing run adaptive_mixed_mode_{i}")
        val_onnx = onnx.load(str(tmpdir / f"adaptive_mixed_mode_{i}.onnx"))
        for j in range(len(gt_onnx.graph.initializer)):
            print(f"Checking initializer {j}")
            gt = gt_onnx.graph.initializer[j]
            gt = to_array(gt)
            val = val_onnx.graph.initializer[j]
            val = to_array(val)
            # print(gt, val)
            assert np.allclose(gt, val)


# Test RMSProp with different parameters constant / non-constant
def test_adaptive_mixed_mode_0(tmpdir):

    # optimizer parameters
    defaultLearningRate = 0.005
    defaultAlpha = 0.8
    defaultMomentum = 0.5
    defaultWeightDecay = 0.1
    defaultEps = 1e-6
    lossScaling = 10.0

    optMaps = [
        {
            0: popart.Adaptive(
                {
                    "defaultLearningRate": (defaultLearningRate, True),
                    "defaultAlpha": (defaultAlpha, True),
                    "defaultMomentum": (defaultMomentum, True),
                    "defaultWeightDecay": (defaultWeightDecay, True),
                    "defaultEps": (defaultEps, True),
                    "lossScaling": (lossScaling, True),
                },
                mode=popart.AdaptiveMode.CenteredRMSProp,
            )
        }
    ]
    outlining = [False]

    for i in range(6):
        optMap = {
            "defaultLearningRate": (defaultLearningRate, True),
            "defaultAlpha": (defaultAlpha, i != 1),
            "defaultMomentum": (defaultMomentum, i != 2),
            "defaultWeightDecay": (defaultWeightDecay, i != 3),
            "defaultEps": (defaultEps, i != 4),
            "lossScaling": (lossScaling, i != 5),
        }
        optMaps = optMaps + [
            {0: popart.Adaptive(optMap, mode=popart.AdaptiveMode.CenteredRMSProp)}
        ]
        outlining = outlining + [False]

    for i in range(6):
        optMap = {
            "defaultLearningRate": (defaultLearningRate, i != 0),
            "defaultAlpha": (defaultAlpha, i != 1),
            "defaultMomentum": (defaultMomentum, i != 2),
            "defaultWeightDecay": (defaultWeightDecay, i != 3),
            "defaultEps": (defaultEps, i != 4),
            "lossScaling": (lossScaling, i != 5),
        }
        optMaps = optMaps + [
            {0: popart.Adaptive(optMap, mode=popart.AdaptiveMode.CenteredRMSProp)}
        ]
        outlining = outlining + [True]

    run_adaptive_mixed_mode(10, optMaps, outlining, tmpdir, np.float32)
    run_adaptive_mixed_mode(10, optMaps, outlining, tmpdir, np.float16)


# Test RMSProp with weight specific const and non-const parameters
def test_adaptive_mixed_mode_1(tmpdir):

    # optimizer parameters
    defaultLearningRate0 = 0.005
    defaultLearningRate5 = 0.0025

    defaultAlpha = 0.7
    defaultMomentum = 0.8
    defaultWeightDecay = 0.1
    defaultEps = 1e-6
    lossScaling = 10.0

    adaptive00 = popart.Adaptive(
        {
            "defaultLearningRate": (defaultLearningRate0, False),
            "defaultAlpha": (defaultAlpha, True),
            "defaultMomentum": (defaultMomentum, True),
            "defaultWeightDecay": (defaultWeightDecay, True),
            "defaultEps": (defaultEps, True),
            "lossScaling": (lossScaling, True),
        }
    )

    adaptive00.insertSpecific("w_0", {"alpha": (0.7, True), "momentum": (0.8, True)})
    adaptive00.insertSpecific("b_0", {"alpha": (0.7, True), "momentum": (0.8, True)})

    adaptive05 = popart.Adaptive(
        {
            "defaultLearningRate": (defaultLearningRate5, False),
            "defaultAlpha": (defaultAlpha, True),
            "defaultMomentum": (defaultMomentum, True),
            "defaultWeightDecay": (defaultWeightDecay, True),
            "defaultEps": (defaultEps, True),
            "lossScaling": (lossScaling, True),
        }
    )

    adaptive05.insertSpecific("w_0", {"alpha": (0.7, True), "momentum": (0.8, True)})
    adaptive05.insertSpecific("b_0", {"alpha": (0.7, True), "momentum": (0.8, True)})

    adaptive10 = popart.Adaptive(
        {
            "defaultLearningRate": (defaultLearningRate0, False),
            "defaultAlpha": (defaultAlpha, False),
            "defaultMomentum": (defaultMomentum, False),
            "defaultWeightDecay": (defaultWeightDecay, False),
            "defaultEps": (defaultEps, False),
            "lossScaling": (lossScaling, False),
        }
    )

    adaptive10.insertSpecific("w_0", {"alpha": (0.7, False), "momentum": (0.8, False)})
    adaptive10.insertSpecific("b_0", {"alpha": (0.7, False), "momentum": (0.8, False)})

    adaptive15 = popart.Adaptive(
        {
            "defaultLearningRate": (defaultLearningRate5, False),
            "defaultAlpha": (defaultAlpha, False),
            "defaultMomentum": (defaultMomentum, False),
            "defaultWeightDecay": (defaultWeightDecay, False),
            "defaultEps": (defaultEps, False),
            "lossScaling": (lossScaling, False),
        }
    )

    adaptive15.insertSpecific("w_0", {"alpha": (0.7, False), "momentum": (0.8, False)})
    adaptive15.insertSpecific("b_0", {"alpha": (0.7, False), "momentum": (0.8, False)})

    adaptive20 = popart.Adaptive(
        {
            "defaultLearningRate": (defaultLearningRate0, False),
            "defaultAlpha": (defaultAlpha, True),
            "defaultMomentum": (defaultMomentum, False),
            "defaultWeightDecay": (defaultWeightDecay, False),
            "defaultEps": (defaultEps, False),
            "lossScaling": (lossScaling, False),
        }
    )

    adaptive20.insertSpecific("w_0", {"alpha": (0.7, False), "momentum": (0.8, True)})
    adaptive20.insertSpecific("b_0", {"alpha": (0.7, False), "momentum": (0.8, True)})

    adaptive25 = popart.Adaptive(
        {
            "defaultLearningRate": (defaultLearningRate5, False),
            "defaultAlpha": (defaultAlpha, True),
            "defaultMomentum": (defaultMomentum, False),
            "defaultWeightDecay": (defaultWeightDecay, False),
            "defaultEps": (defaultEps, False),
            "lossScaling": (lossScaling, False),
        }
    )

    adaptive25.insertSpecific("w_0", {"alpha": (0.7, False), "momentum": (0.8, True)})
    adaptive25.insertSpecific("b_0", {"alpha": (0.7, False), "momentum": (0.8, True)})

    # Change RMSProp optimizer after 0 and 5 steps
    optMaps = [
        {0: adaptive00, 5: adaptive05},
        {0: adaptive10, 5: adaptive15},
        {0: adaptive20, 5: adaptive25},
    ]

    outlining = [True, True, True]

    run_adaptive_mixed_mode(10, optMaps, outlining, tmpdir, np.float32)

    debug_filename = str(tmpdir) + "/debug.json"
    popart.initializePoplarDebugInfo(debug_filename, "json")

    run_adaptive_mixed_mode(10, optMaps, outlining, tmpdir, np.float16)

    # Ensure adaptive operations have correct debug context
    popart.closePoplarDebugInfo()
    num_adaptives = 0
    parents = set()
    with open(debug_filename, encoding="utf-8") as json_file:
        data = json.load(json_file)
        for context in data["contexts"]:
            if (
                context["layer"] == "popart"
                and "opid" in context
                and "Adaptive" in context["opid"]
            ):
                parents.add(context["parentId"])
                num_adaptives += 1
        for context in data["contexts"]:
            if context["id"] in parents:
                parents.remove(context["id"])
                assert context["layer"] == "popart_builder"
                assert data["stringTable"][context["name"]] == "adaptive"
    assert num_adaptives == 24
    assert len(parents) == 0
