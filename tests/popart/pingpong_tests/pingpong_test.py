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


def run_model(tmpdir,
              model_file_name,
              enable_pingpong=False,
              enable_matmul_serialization=False,
              enable_outlining=False,
              num_layers=3,
              dsize=48,
              batch_size=1,
              num_iterations=1,
              num_replicas=1,
              replicated_weight_sharding=False,
              optimizer_dict={"defaultLearningRate": (0.5, False)}):

    np.random.seed(10911)
    matmul_serialization_mode = 'output_channels'
    matmul_serialization_factor = 2

    builder = popart.Builder()
    ip = builder.addInputTensor(
        popart.TensorInfo("FLOAT", [batch_size, dsize, dsize]))

    def add_layer(index, in_id):
        w = builder.addInitializedInputTensor(
            np.random.rand(dsize, dsize).astype(np.float32), f"W{index}")
        matmul_id = builder.aiOnnx.matmul([in_id, w])
        if enable_matmul_serialization:
            builder.setSerializeMatMul({matmul_id}, matmul_serialization_mode,
                                       matmul_serialization_factor)
        return matmul_id

    out = ip
    for i in range(num_layers):
        with builder.pingPongPhase(i):
            out = add_layer(i, out)

    l1 = builder.aiGraphcore.l1loss([out], 0.1)

    anchorIds = []

    builder.addOutputTensor(out)

    device = tu.create_test_device(
        num_replicas * (2 if enable_pingpong else 1),
        1216,
        None,
        pattern=popart.SyncPattern.Full)

    dfAnchors = {}
    for anchorId in anchorIds:
        dfAnchors.update({anchorId: popart.AnchorReturnType("All")})

    opts = popart.SessionOptions()
    opts.enableOutlining = enable_outlining
    opts.enableReplicatedGraphs = True if num_replicas > 1 else False
    opts.replicatedGraphCount = num_replicas
    opts.replicatedWeightSharding = replicated_weight_sharding
    opts.replicatedWeightShardingMinNumElements = 0

    optimizer = popart.SGD(optimizer_dict)

    if (enable_pingpong):
        opts.pingPongPhases = num_layers
        opts.autoRecomputation = popart.RecomputationType.NoRecompute
        opts.virtualGraphMode = popart.VirtualGraphMode.PingPong
        opts.explicitRecomputation = False

    proto = builder.getModelProto()

    session = popart.TrainingSession(fnModel=proto,
                                     dataFlow=popart.DataFlow(1, dfAnchors),
                                     optimizer=optimizer,
                                     loss=l1,
                                     patterns=popart.Patterns(
                                         popart.PatternsLevel.All),
                                     userOptions=opts,
                                     deviceInfo=device)

    session.prepareDevice()
    session.weightsFromHost()
    anchors = session.initAnchorArrays()

    for i in range(num_iterations):
        ip_data = np.random.rand(num_replicas, batch_size, dsize,
                                 dsize).astype(np.float32)
        stepio = popart.PyStepIO({ip: ip_data}, anchors)
        session.run(stepio)

    print("anchors:")
    print(anchors)

    session.modelToHost(str(tmpdir / model_file_name))


def check_model(lhs_model, rhs_model):
    for i in range(len(lhs_model.graph.initializer)):
        print(f'Checking initializer {i}')
        lhs = lhs_model.graph.initializer[i]
        lhs = numpy_helper.to_array(lhs)
        rhs = rhs_model.graph.initializer[i]
        rhs = numpy_helper.to_array(rhs)
        assert np.allclose(lhs, rhs)


@tu.requires_ipu
def test_weight_update(tmpdir):

    run_model(tmpdir, 'without_pingpong.onnx', False, True)
    run_model(tmpdir, 'with_pingpong.onnx', True, False)
    run_model(tmpdir, 'with_pingpong_serialized.onnx', True, True)

    without_pingpong = onnx.load(str(tmpdir / 'without_pingpong.onnx'))
    with_pingpong = onnx.load(str(tmpdir / 'with_pingpong.onnx'))
    with_pingpong_serialized = onnx.load(
        str(tmpdir / 'with_pingpong_serialized.onnx'))

    check_model(without_pingpong, with_pingpong)
    check_model(without_pingpong, with_pingpong_serialized)


# Check that 2 batches on 1 replica or 1 batch per replica on 2 replicas
# results in the same updated weight with SGD0
@tu.requires_ipu
def test_replicated_sgd0_weight_update(tmpdir):

    run_model(tmpdir,
              'pingpong.onnx',
              enable_pingpong=True,
              batch_size=2,
              num_replicas=1)
    run_model(tmpdir,
              'pingpong_replicated.onnx',
              enable_pingpong=True,
              batch_size=1,
              num_replicas=2)
    run_model(tmpdir,
              'pingpong_replicated_rws.onnx',
              enable_pingpong=True,
              batch_size=1,
              num_replicas=2,
              replicated_weight_sharding=True)

    pingpong = onnx.load(str(tmpdir / 'pingpong.onnx'))
    pingpong_replicated = onnx.load(str(tmpdir / 'pingpong_replicated.onnx'))
    pingpong_replicated_rws = onnx.load(
        str(tmpdir / 'pingpong_replicated_rws.onnx'))

    check_model(pingpong, pingpong_replicated)
    check_model(pingpong, pingpong_replicated_rws)


# Check that 2 batches on 1 replica or 1 batch per replica on 2 replicas
# results in the same updated weight with SGD1
@tu.requires_ipu
def test_replicated_sgd1_weight_update(tmpdir):

    optimizer_dict = {
        "defaultLearningRate": (0.1, False),
        "defaultMomentum": (0.9, False),
        "defaultDampening": (0.2, False),
        "defaultVelocityScaling": (0.1, False),
        "lossScaling": (1.0, True),
        "defaultWeightDecay": (0.2, True)
    }

    run_model(tmpdir,
              'pingpong.onnx',
              enable_pingpong=True,
              batch_size=2,
              num_replicas=1,
              num_iterations=5,
              optimizer_dict=optimizer_dict)
    run_model(tmpdir,
              'pingpong_replicated.onnx',
              enable_pingpong=True,
              batch_size=1,
              num_replicas=2,
              num_iterations=5,
              optimizer_dict=optimizer_dict)
    run_model(tmpdir,
              'pingpong_replicated_rws.onnx',
              enable_pingpong=True,
              batch_size=1,
              num_replicas=2,
              num_iterations=5,
              optimizer_dict=optimizer_dict,
              replicated_weight_sharding=True)

    pingpong = onnx.load(str(tmpdir / 'pingpong.onnx'))
    pingpong_replicated = onnx.load(str(tmpdir / 'pingpong_replicated.onnx'))
    pingpong_replicated_rws = onnx.load(
        str(tmpdir / 'pingpong_replicated_rws.onnx'))

    check_model(pingpong, pingpong_replicated)
    check_model(pingpong, pingpong_replicated_rws)
