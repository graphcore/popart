# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import numpy as np
import pytest
import popart
import onnx
from onnx import numpy_helper
from onnx import TensorProto

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


def run_sgd_mixed_mode(steps,
                       opt_dicts,
                       enable_outlining,
                       tmpdir,
                       dtype=np.float32):
    def run(opt_dict, enable_outlining, model_file_name):
        np.random.seed(1878)
        dsize = 10
        builder = popart.Builder()
        ip = builder.addInputTensor(
            popart.TensorInfo("FLOAT" if dtype == np.float32 else "FLOAT16",
                              [dsize, dsize]))
        d__ip = popart.reservedGradientPrefix() + ip

        def add_layer(in_id, name):
            w = builder.addInitializedInputTensor(
                np.random.rand(dsize, dsize).astype(dtype), "w_" + name)
            b = builder.addInitializedInputTensor(
                np.random.rand(dsize).astype(dtype), "b_" + name)
            matmul_id = builder.aiOnnx.gemm([in_id, w, b], 1, 1, False, False)
            return matmul_id

        m1 = add_layer(ip, "0")
        m2 = add_layer(m1, "1")
        m3 = add_layer(m2, "2")
        m4 = add_layer(m3, "3")

        out = builder.aiGraphcore.identityloss([m4])
        builder.addOutputTensor(out)

        device = tu.create_test_device()

        anchors = {}

        opts = popart.SessionOptions()
        opts.enableOutliningCopyCostPruning = False
        opts.outlineThreshold = -np.inf
        opts.enableOutlining = enable_outlining

        proto = builder.getModelProto()

        session = popart.TrainingSession(fnModel=proto,
                                         dataFlow=popart.DataFlow(1, anchors),
                                         optimizer=opt_dict[0],
                                         loss=out,
                                         patterns=popart.Patterns(
                                             popart.PatternsLevel.All),
                                         userOptions=opts,
                                         deviceInfo=device)

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
        print(f"Running sgd_mixed_mode_{i}")
        run(opt_dict, enable_outlining[i], f"sgd_mixed_mode_{i}.onnx")

    gt_onnx = onnx.load(str(tmpdir / "sgd_mixed_mode_0.onnx"))

    for i, opt_dict in enumerate(opt_dicts):
        print(f"Testing run sgd_mixed_mode_{i}")
        val_onnx = onnx.load(str(tmpdir / f"sgd_mixed_mode_{i}.onnx"))
        for j in range(len(gt_onnx.graph.initializer)):
            print(f"Checking initializer {j}")
            gt = gt_onnx.graph.initializer[j]
            gt = to_array(gt)
            val = val_onnx.graph.initializer[j]
            val = to_array(val)
            # print(gt, val)
            assert np.allclose(gt, val, rtol=1.e-2, atol=1.e-3)


TENSOR_TYPES = [popart.DataType.FLOAT, popart.DataType.FLOAT16]


# Test SGD with different parameters constant / non-constant
@pytest.mark.parametrize("sgdAccMm", [
    popart.SGDAccumulatorAndMomentum.Separate,
    popart.SGDAccumulatorAndMomentum.Combined
])
@pytest.mark.parametrize("accumType", TENSOR_TYPES)
@pytest.mark.parametrize("accl1Type", TENSOR_TYPES)
def test_sgd_mixed_mode_0(tmpdir, sgdAccMm, accumType, accl1Type):

    if sgdAccMm == popart.SGDAccumulatorAndMomentum.Combined:
        # types are ignored so skip if not Float, Float
        if (accumType != popart.DataType.FLOAT
                or accl1Type != popart.DataType.FLOAT):
            return

    #optimizer parameters
    defaultLearningRate = 1e-4
    defaultMomentum = 0.7
    defaultVelocityScaling = 1.0
    defaultWeightDecay = 0.1
    defaultDampening = 0.05
    lossScaling = 10.0

    optMaps = [{
        0:
        popart.SGD(
            {
                "defaultLearningRate": (defaultLearningRate, True),
                "defaultMomentum": (defaultMomentum, True),
                "defaultVelocityScaling": (defaultVelocityScaling, True),
                "defaultWeightDecay": (defaultWeightDecay, True),
                "defaultDampening": (defaultDampening, True),
                "lossScaling": (lossScaling, True),
            },
            accumulatorAndMomentum=sgdAccMm,
            accumType=accumType,
            accl1Type=accl1Type)
    }]
    outlining = [False]

    for i in range(6):
        optMap = {
            "defaultLearningRate": (defaultLearningRate, i != 0),
            "defaultMomentum": (defaultMomentum, i != 1),
            "defaultVelocityScaling": (defaultVelocityScaling, i != 2),
            "defaultWeightDecay": (defaultWeightDecay, i != 3),
            "defaultDampening": (defaultDampening, i != 4),
            "lossScaling": (lossScaling, i != 5),
        }
        optMaps = optMaps + [{
            0:
            popart.SGD(optMap,
                       accumulatorAndMomentum=sgdAccMm,
                       accumType=accumType,
                       accl1Type=accl1Type)
        }]
        outlining = outlining + [False]

    for i in range(6):
        optMap = {
            "defaultLearningRate": (defaultLearningRate, i != 0),
            "defaultMomentum": (defaultMomentum, i != 1),
            "defaultVelocityScaling": (defaultVelocityScaling, i != 2),
            "defaultWeightDecay": (defaultWeightDecay, i != 3),
            "defaultDampening": (defaultDampening, i != 4),
            "lossScaling": (lossScaling, i != 5),
        }
        optMaps = optMaps + [{
            0:
            popart.SGD(optMap,
                       accumulatorAndMomentum=sgdAccMm,
                       accumType=accumType,
                       accl1Type=accl1Type)
        }]
        outlining = outlining + [True]

    run_sgd_mixed_mode(10, optMaps, outlining, tmpdir, np.float32)
    run_sgd_mixed_mode(10, optMaps, outlining, tmpdir, np.float16)


# Test SGD with weight specific const and non-const parameters
def test_sgd_mixed_mode_1(tmpdir):

    #optimizer parameters
    defaultLearningRate0 = 2e-4
    defaultLearningRate5 = 1e-4

    defaultMomentum = 0.7
    defaultVelocityScaling = 0.8
    defaultWeightDecay = 0.1
    defaultDampening = 0.05
    lossScaling = 10.0

    sgd00 = popart.SGD({
        "defaultLearningRate": (defaultLearningRate0, False),
        "defaultMomentum": (defaultMomentum, True),
        "defaultVelocityScaling": (defaultVelocityScaling, True),
        "defaultWeightDecay": (defaultWeightDecay, True),
        "defaultDampening": (defaultDampening, True),
        "lossScaling": (lossScaling, True),
    })

    sgd00.insertSpecific("w_0", {
        "momentum": (0.9, True),
        "velocityScaling": (0.25, True)
    })
    sgd00.insertSpecific("b_0", {
        "momentum": (0.9, True),
        "velocityScaling": (0.25, True)
    })

    sgd05 = popart.SGD({
        "defaultLearningRate": (defaultLearningRate5, False),
        "defaultMomentum": (defaultMomentum, True),
        "defaultVelocityScaling": (defaultVelocityScaling, True),
        "defaultWeightDecay": (defaultWeightDecay, True),
        "defaultDampening": (defaultDampening, True),
        "lossScaling": (lossScaling, True),
    })

    sgd05.insertSpecific("w_0", {
        "momentum": (0.9, True),
        "velocityScaling": (0.25, True)
    })
    sgd05.insertSpecific("b_0", {
        "momentum": (0.9, True),
        "velocityScaling": (0.25, True)
    })

    sgd10 = popart.SGD({
        "defaultLearningRate": (defaultLearningRate0, False),
        "defaultMomentum": (defaultMomentum, False),
        "defaultVelocityScaling": (defaultVelocityScaling, False),
        "defaultWeightDecay": (defaultWeightDecay, False),
        "defaultDampening": (defaultDampening, False),
        "lossScaling": (lossScaling, False),
    })

    sgd10.insertSpecific("w_0", {
        "momentum": (0.9, False),
        "velocityScaling": (0.25, False)
    })
    sgd10.insertSpecific("b_0", {
        "momentum": (0.9, False),
        "velocityScaling": (0.25, False)
    })

    sgd15 = popart.SGD({
        "defaultLearningRate": (defaultLearningRate5, False),
        "defaultMomentum": (defaultMomentum, False),
        "defaultVelocityScaling": (defaultVelocityScaling, False),
        "defaultWeightDecay": (defaultWeightDecay, False),
        "defaultDampening": (defaultDampening, False),
        "lossScaling": (lossScaling, False),
    })

    sgd15.insertSpecific("w_0", {
        "momentum": (0.9, False),
        "velocityScaling": (0.25, False)
    })
    sgd15.insertSpecific("b_0", {
        "momentum": (0.9, False),
        "velocityScaling": (0.25, False)
    })

    sgd20 = popart.SGD({
        "defaultLearningRate": (defaultLearningRate0, False),
        "defaultMomentum": (defaultMomentum, True),
        "defaultVelocityScaling": (defaultVelocityScaling, False),
        "defaultWeightDecay": (defaultWeightDecay, False),
        "defaultDampening": (defaultDampening, False),
        "lossScaling": (lossScaling, False),
    })

    sgd20.insertSpecific("w_0", {
        "momentum": (0.9, False),
        "velocityScaling": (0.25, True)
    })
    sgd20.insertSpecific("b_0", {
        "momentum": (0.9, False),
        "velocityScaling": (0.25, True)
    })

    sgd25 = popart.SGD({
        "defaultLearningRate": (defaultLearningRate5, False),
        "defaultMomentum": (defaultMomentum, True),
        "defaultVelocityScaling": (defaultVelocityScaling, False),
        "defaultWeightDecay": (defaultWeightDecay, False),
        "defaultDampening": (defaultDampening, False),
        "lossScaling": (lossScaling, False),
    })

    sgd25.insertSpecific("w_0", {
        "momentum": (0.9, False),
        "velocityScaling": (0.25, True)
    })
    sgd25.insertSpecific("b_0", {
        "momentum": (0.9, False),
        "velocityScaling": (0.25, True)
    })

    # Change SGD optimizer after 0 and 5 steps
    optMaps = [{
        0: sgd00,
        5: sgd05
    }, {
        0: sgd10,
        5: sgd15
    }, {
        0: sgd20,
        5: sgd25
    }]

    outlining = [True, True, True]

    run_sgd_mixed_mode(10, optMaps, outlining, tmpdir, np.float32)
    run_sgd_mixed_mode(10, optMaps, outlining, tmpdir, np.float16)
