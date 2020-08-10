# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import os
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
              enable_executionphases=False,
              enable_matmul_serialization=False,
              enable_outlining=False,
              activation_tensor_location_settings=None,
              weight_tensor_location_settings=None,
              optimizer_state_tensor_location_settings=None,
              num_layers=3,
              dsize=48,
              batch_size=1,
              num_iterations=1,
              num_replicas=1,
              replicated_tensor_sharding=False,
              optimizer=popart.SGD({"defaultLearningRate": (0.5, False)})):

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
        with builder.executionPhase(i):
            out = add_layer(i, out)

    l1 = builder.aiGraphcore.l1loss([out], 0.1)

    anchorIds = []

    builder.addOutputTensor(out)

    device = tu.create_test_device(
        num_replicas * (2 if enable_executionphases else 1),
        pattern=popart.SyncPattern.Full)

    dfAnchors = {}
    for anchorId in anchorIds:
        dfAnchors.update({anchorId: popart.AnchorReturnType("All")})

    opts = popart.SessionOptions()
    opts.enableOutlining = enable_outlining
    opts.enableReplicatedGraphs = True if num_replicas > 1 else False
    opts.replicatedGraphCount = num_replicas

    if activation_tensor_location_settings is not None:
        opts.activationTensorLocationSettings = activation_tensor_location_settings
    if weight_tensor_location_settings is not None:
        opts.weightTensorLocationSettings = weight_tensor_location_settings
    if optimizer_state_tensor_location_settings is not None:
        opts.optimizerStateTensorLocationSettings = optimizer_state_tensor_location_settings
    if (enable_executionphases):
        opts.executionPhaseSettings.phases = num_layers
        opts.autoRecomputation = popart.RecomputationType.NoRecompute
        opts.virtualGraphMode = popart.VirtualGraphMode.ExecutionPhases
        opts.explicitRecomputation = False

    opts.weightTensorLocationSettings.minElementsForReplicatedTensorSharding = 8
    opts.weightTensorLocationSettings.location.replicatedTensorSharding = replicated_tensor_sharding
    opts.optimizerStateTensorLocationSettings.minElementsForReplicatedTensorSharding = 8
    opts.optimizerStateTensorLocationSettings.location.replicatedTensorSharding = replicated_tensor_sharding

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
        assert np.allclose(lhs, rhs, rtol=1.e-4, atol=1.e-6)


@tu.requires_ipu
def test_weight_update(tmpdir):

    run_model(tmpdir, 'without_phased.onnx', False, True)
    run_model(tmpdir, 'with_phased.onnx', True, False)
    run_model(tmpdir, 'with_phased_serialized.onnx', True, True)

    without_phased = onnx.load(str(tmpdir / 'without_phased.onnx'))
    with_phased = onnx.load(str(tmpdir / 'with_phased.onnx'))
    with_phased_serialized = onnx.load(
        str(tmpdir / 'with_phased_serialized.onnx'))

    check_model(without_phased, with_phased)
    check_model(without_phased, with_phased_serialized)


@tu.requires_ipu
def test_onchip_memory(tmpdir):
    onchip_settings = popart.TensorLocationSettings(
        popart.TensorStorage.OnChip, 0)
    run_model(tmpdir, 'model_normal.onnx', enable_executionphases=False)
    run_model(tmpdir,
              'model_onchip_act.onnx',
              enable_executionphases=True,
              activation_tensor_location_settings=onchip_settings)
    run_model(tmpdir,
              'model_onchip_weights.onnx',
              enable_executionphases=True,
              weight_tensor_location_settings=onchip_settings)
    run_model(tmpdir,
              'model_onchip_opt_state.onnx',
              enable_executionphases=True,
              optimizer_state_tensor_location_settings=onchip_settings)

    normal = onnx.load(str(tmpdir / 'model_normal.onnx'))
    onchip_act = onnx.load(str(tmpdir / 'model_onchip_act.onnx'))
    onchip_weights = onnx.load(str(tmpdir / 'model_onchip_weights.onnx'))
    onchip_opt_state = onnx.load(str(tmpdir / 'model_onchip_opt_state.onnx'))

    check_model(normal, onchip_act)
    check_model(normal, onchip_weights)
    check_model(normal, onchip_opt_state)


@tu.requires_ipu
def test_inplacing_phased_constraints(tmpdir):
    # This used to fail, see T23985
    run_model(tmpdir,
              'phased.onnx',
              enable_executionphases=True,
              num_layers=5,
              optimizer=popart.SGD({
                  "defaultLearningRate": (0.1, True),
                  "defaultMomentum": (0.0, False),
                  "defaultWeightDecay": (0.0, False),
                  "defaultDampening": (0.0, True)
              }))


# Check that 2 batches on 1 replica or 1 batch per replica on 2 replicas
# results in the same updated weight with SGD0
@tu.requires_ipu
def test_replicated_sgd0_weight_update(tmpdir):

    run_model(tmpdir,
              'phased.onnx',
              enable_executionphases=True,
              batch_size=2,
              num_replicas=1)
    run_model(tmpdir,
              'phased_replicated.onnx',
              enable_executionphases=True,
              batch_size=1,
              num_replicas=2)
    run_model(tmpdir,
              'phased_replicated_rws.onnx',
              enable_executionphases=True,
              batch_size=1,
              num_replicas=2,
              replicated_tensor_sharding=True)

    phased = onnx.load(str(tmpdir / 'phased.onnx'))
    phased_replicated = onnx.load(str(tmpdir / 'phased_replicated.onnx'))
    phased_replicated_rws = onnx.load(
        str(tmpdir / 'phased_replicated_rws.onnx'))

    check_model(phased, phased_replicated)
    check_model(phased, phased_replicated_rws)


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
              'phased.onnx',
              enable_executionphases=True,
              batch_size=2,
              num_replicas=1,
              num_iterations=5,
              optimizer=popart.SGD(optimizer_dict))
    run_model(tmpdir,
              'phased_replicated.onnx',
              enable_executionphases=True,
              batch_size=1,
              num_replicas=2,
              num_iterations=5,
              optimizer=popart.SGD(optimizer_dict))
    run_model(tmpdir,
              'phased_replicated_rws.onnx',
              enable_executionphases=True,
              batch_size=1,
              num_replicas=2,
              num_iterations=5,
              optimizer=popart.SGD(optimizer_dict),
              replicated_tensor_sharding=True)

    phased = onnx.load(str(tmpdir / 'phased.onnx'))
    phased_replicated = onnx.load(str(tmpdir / 'phased_replicated.onnx'))
    phased_replicated_rws = onnx.load(
        str(tmpdir / 'phased_replicated_rws.onnx'))

    check_model(phased, phased_replicated)
    check_model(phased, phased_replicated_rws)


# Check that 2 batches on 1 replica or 1 batch per replica on 2 replicas
# results in the same updated weight with SGD1
@tu.requires_ipu
def test_replicated_adam_weight_update(tmpdir):

    optimizer_dict = {
        "defaultLearningRate": (0.005, True),
        "defaultBeta1": (0.7, True),
        "defaultBeta2": (0.8, True),
        "defaultWeightDecay": (0.1, True),
        "defaultEps": (1e-6, True),
        "lossScaling": (10.0, True),
    }

    run_model(tmpdir,
              'phased.onnx',
              enable_executionphases=True,
              batch_size=2,
              num_replicas=1,
              num_iterations=5,
              optimizer=popart.Adam(optimizer_dict))
    run_model(tmpdir,
              'phased_replicated.onnx',
              enable_executionphases=True,
              batch_size=1,
              num_replicas=2,
              num_iterations=5,
              optimizer=popart.Adam(optimizer_dict))
    run_model(tmpdir,
              'phased_replicated_rws.onnx',
              enable_executionphases=True,
              batch_size=1,
              num_replicas=2,
              num_iterations=5,
              optimizer=popart.Adam(optimizer_dict),
              replicated_tensor_sharding=True)

    phased = onnx.load(str(tmpdir / 'phased.onnx'))
    phased_replicated = onnx.load(str(tmpdir / 'phased_replicated.onnx'))
    phased_replicated_rws = onnx.load(
        str(tmpdir / 'phased_replicated_rws.onnx'))

    check_model(phased, phased_replicated)
    check_model(phased, phased_replicated_rws)


# Check that 2 batches on 1 replica or 1 batch per replica on 2 replicas
# results in the same updated weight with SGD1
@tu.requires_ipu
def test_replicated_lamb_weight_update(tmpdir):

    optimizer_dict = {
        "defaultLearningRate": (0.005, True),
        "defaultBeta1": (0.7, True),
        "defaultBeta2": (0.8, True),
        "defaultWeightDecay": (0.1, True),
        "defaultEps": (1e-6, True),
        "lossScaling": (10.0, True),
    }

    run_model(tmpdir,
              'phased.onnx',
              enable_executionphases=True,
              batch_size=2,
              num_replicas=1,
              num_iterations=5,
              optimizer=popart.Adam(optimizer_dict, popart.AdamMode.Lamb))
    run_model(tmpdir,
              'phased_replicated.onnx',
              enable_executionphases=True,
              batch_size=1,
              num_replicas=2,
              num_iterations=5,
              optimizer=popart.Adam(optimizer_dict, popart.AdamMode.Lamb))
    run_model(tmpdir,
              'phased_replicated_rws.onnx',
              enable_executionphases=True,
              batch_size=1,
              num_replicas=2,
              num_iterations=5,
              optimizer=popart.Adam(optimizer_dict, popart.AdamMode.Lamb),
              replicated_tensor_sharding=True)

    phased = onnx.load(str(tmpdir / 'phased.onnx'))
    phased_replicated = onnx.load(str(tmpdir / 'phased_replicated.onnx'))
    phased_replicated_rws = onnx.load(
        str(tmpdir / 'phased_replicated_rws.onnx'))

    check_model(phased, phased_replicated)
    check_model(phased, phased_replicated_rws)
