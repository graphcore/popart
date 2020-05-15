# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import popart
import numpy as np
import torch
import onnx
from onnx import numpy_helper

# `import test_util` requires adding to sys.path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu


@tu.requires_ipu
def test_weight_update(tmpdir):
    def run(model_file_name, enablePingpong, enableMatMulSerialization):
        np.random.seed(10911)
        num_layers = 3
        dsize = 4
        matmul_serialization_mode = 'output_channels'
        matmul_serialization_factor = 2

        builder = popart.Builder()
        ip = builder.addInputTensor(popart.TensorInfo("FLOAT", [dsize, dsize]))

        def add_layer(index, in_id):
            w = builder.addInitializedInputTensor(
                np.random.rand(dsize, dsize).astype(np.float32), f"W{index}")
            matmul_id = builder.aiOnnx.matmul([in_id, w])
            if enableMatMulSerialization:
                builder.setSerializeMatMul({matmul_id},
                                           matmul_serialization_mode,
                                           matmul_serialization_factor)
            return matmul_id

        out = ip
        for i in range(num_layers):
            with builder.pingPongPhase(i):
                out = add_layer(i, out)

        l1 = builder.aiGraphcore.l1loss([out], 0.1)

        anchorIds = []

        builder.addOutputTensor(out)

        device = tu.create_test_device(2 if enablePingpong else 1,
                                       1216,
                                       None,
                                       pattern=popart.SyncPattern.Full)

        dfAnchors = {}
        for anchorId in anchorIds:
            dfAnchors.update({anchorId: popart.AnchorReturnType("All")})

        opts = popart.SessionOptions()
        opts.enableOutlining = False
        if (enablePingpong):
            opts.pingPongPhases = num_layers
            opts.autoRecomputation = popart.RecomputationType.NoRecompute
            opts.virtualGraphMode = popart.VirtualGraphMode.PingPong
            opts.explicitRecomputation = False

        proto = builder.getModelProto()

        session = popart.TrainingSession(
            fnModel=proto,
            dataFlow=popart.DataFlow(1, dfAnchors),
            optimizer=popart.ConstSGD(0.5),
            losses=[popart.IdentityLoss(l1, "l1LossVal")],
            patterns=popart.Patterns(popart.PatternsLevel.All),
            userOptions=opts,
            deviceInfo=device)

        session.prepareDevice()
        session.weightsFromHost()
        anchors = session.initAnchorArrays()

        ip_data = np.random.rand(dsize, dsize).astype(np.float32)
        stepio = popart.PyStepIO({ip: ip_data}, anchors)

        session.run(stepio)

        print("anchors:")
        print(anchors)

        session.modelToHost(str(tmpdir / model_file_name))

    run('without_pingpong.onnx', False, True)
    run('with_pingpong.onnx', True, False)
    run('with_pingpong_serialized.onnx', True, True)

    without_pingpong = onnx.load(str(tmpdir / 'without_pingpong.onnx'))
    with_pingpong = onnx.load(str(tmpdir / 'with_pingpong.onnx'))
    with_pingpong_serialized = onnx.load(
        str(tmpdir / 'with_pingpong_serialized.onnx'))

    for i in range(len(without_pingpong.graph.initializer)):
        print(f'Checking initializer {i} for pingpong')
        lhs = without_pingpong.graph.initializer[i]
        lhs = numpy_helper.to_array(lhs)
        rhs = with_pingpong.graph.initializer[i]
        rhs = numpy_helper.to_array(rhs)
        assert np.allclose(lhs, rhs)

    for i in range(len(without_pingpong.graph.initializer)):
        print(
            f'Checking initializer {i} for pingpong and matmul serialisation')
        lhs = without_pingpong.graph.initializer[i]
        lhs = numpy_helper.to_array(lhs)
        rhs = with_pingpong_serialized.graph.initializer[i]
        rhs = numpy_helper.to_array(rhs)
        assert np.allclose(lhs, rhs)
