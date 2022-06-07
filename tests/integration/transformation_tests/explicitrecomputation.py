# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart
import numpy as np
import onnx
import json
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

        with tu.create_test_device() as device:

            dataflow_anchors = {}
            for anchorId in anchorIds:
                dataflow_anchors.update(
                    {anchorId: popart.AnchorReturnType("All")})

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


def test_explicit_recomputation_pipelining():
    """ Test that pipeline recomputation recomputes as expected """
    batches_per_step = 3
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
        act_id = builder.aiOnnxOpset10.relu([matmul_id])
        return act_id

    with builder.virtualGraph(0), builder.pipelineStage(0):
        m1 = add_layer(ip)
        m2 = add_layer(m1)
    with builder.virtualGraph(1), builder.pipelineStage(1):
        m3 = add_layer(m2)
        m3 = add_layer(m3)
        out = builder.aiGraphcore.identityloss([m3])

    anchorIds = [out]
    builder.addOutputTensor(out)

    with tu.create_test_device(numIpus=2, opts={"compileIPUCode":
                                                False}) as device:

        dataflow_anchors = {}
        for anchorId in anchorIds:
            dataflow_anchors.update({anchorId: popart.AnchorReturnType("All")})

        opts = popart.SessionOptions()
        opts.enablePipelining = True
        opts.enableExplicitIR(True)
        opts.enableOutlining = False
        opts.virtualGraphMode = popart.VirtualGraphMode.Manual
        opts.autoRecomputation = popart.RecomputationType.Pipeline

        proto = builder.getModelProto()

        session = popart.TrainingSession(
            fnModel=proto,
            dataFlow=popart.DataFlow(batches_per_step, dataflow_anchors),
            optimizer=popart.ConstSGD(0.01),
            loss=out,
            patterns=popart.Patterns(popart.PatternsLevel.All),
            userOptions=opts,
            deviceInfo=device)

        session.prepareDevice()
        session.weightsFromHost()
        anchors = session.initAnchorArrays()

        ip_data = np.ones((batches_per_step, dsize, dsize), dtype=np.float32)
        stepio = popart.PyStepIO({ip: ip_data}, anchors)

        session.run(stepio)

        ir = json.loads(session._serializeIr(
            popart.IrSerializationFormat.JSON))

        stashes = [
            op for graph in ir for op in ir[graph]
            if (op["type"] == "DynamicUpdate"
                or op["type"] == "DynamicUpdateInplace")
        ]

        # Due to the number of pipeline stages and only one tensor requring
        # stashing in pipeline stage 0, we expect one stash
        assert len(stashes) == 1

        recomputed = [
            op for graph in ir for op in ir[graph]
            if (op["attributes"]["recomputetype"] == "Recomputed")
        ]

        # Due to the number of operations in the original forward pass,
        # we expect 6 recomputed operations (when outlining is disabled)
        # This may change with T61001
        assert len(recomputed) == 6
