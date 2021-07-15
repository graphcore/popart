import popart
import numpy as np
import torch
import onnx
from onnx import numpy_helper

# 'import test_util' requires adding to sys.path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu


def test_explicit_recomputation(tmpdir):
    def run(model_file_name, explicit_recompute=True):
        dsize = 10
        builder = popart.Builder()
        ip = builder.addInputTensor(popart.TensorInfo("FLOAT", [dsize, dsize]))
        d__ip = popart.reservedGradientPrefix() + ip

        def add_layer(in_id):
            np.random.seed(1)
            scaler = 0.01
            w = builder.addInitializedInputTensor(
                np.random.randn(dsize, dsize).astype(np.float32) * scaler)
            b = builder.addInitializedInputTensor(
                np.zeros((dsize, 1)).astype(np.float32))
            matmul_id = builder.aiOnnxOpset10.gemm([in_id, w, b])
            return matmul_id

        if explicit_recompute:
            with builder.recomputeOutput(popart.RecomputeType.Recompute):
                m1 = add_layer(ip)
                m2 = add_layer(m1)
                m3 = add_layer(m2)
        else:
            m1 = add_layer(ip)
            m2 = add_layer(m1)
            m3 = add_layer(m2)

        anchorIds = []
        for i in (ip, m1, m2, m3):
            anchorIds.append(popart.reservedGradientPrefix() + i)

        out = builder.aiGraphcore.identityloss([m3])
        builder.addOutputTensor(out)

        device = tu.create_test_device()

        dataflow_anchors = {}
        for anchorId in anchorIds:
            dataflow_anchors.update({anchorId: popart.AnchorReturnType("All")})

        opts = popart.SessionOptions()
        opts.explicitRecomputation = explicit_recompute

        proto = builder.getModelProto()

        session = popart.TrainingSession(
            fnModel=proto,
            dataFlow=popart.DataFlow(1, dataflow_anchors),
            optimizer=popart.ConstSGD(0.01),
            loss=out,
            patterns=popart.Patterns(popart.PatternsLevel.All),
            userOptions=opts,
            deviceInfo=device)

        session.prepareDevice()
        session.weightsFromHost()
        anchors = session.initAnchorArrays()

        ip_data = np.ones((dsize, dsize), dtype=np.float32)
        stepio = popart.PyStepIO({ip: ip_data}, anchors)

        session.run(stepio)
        session.modelToHost(str(tmpdir / model_file_name))

    run("explicit_recomputation_disabled.onnx", False)
    run("explicit_recomputation_enabled.onnx", True)

    explicit_recomputation_disabled = onnx.load(
        str(tmpdir / "explicit_recomputation_disabled.onnx"))
    explicit_recomputation_enabled = onnx.load(
        str(tmpdir / "explicit_recomputation_enabled.onnx"))

    for i in range(len(explicit_recomputation_disabled.graph.initializer)):
        print(f"Checking initializer {i}")
        lhs = explicit_recomputation_disabled.graph.initializer[i]
        lhs = numpy_helper.to_array(lhs)
        rhs = explicit_recomputation_enabled.graph.initializer[i]
        rhs = numpy_helper.to_array(rhs)
        assert np.allclose(lhs, rhs)
