# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import json
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


def get_ir(model_file_name='model.onnx',
           enable_pingpong=True,
           enable_matmul_serialization=False,
           enable_outlining=False,
           activation_tensor_location_settings=None,
           weight_tensor_location_settings=None,
           optimizer_state_tensor_location_settings=None,
           tensor_location_setting_override={},
           num_layers=3,
           dsize=48,
           batch_size=1,
           num_iterations=1,
           num_replicas=1,
           replicated_weight_sharding=False,
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
        with builder.pingPongPhase(i):
            out = add_layer(i, out)

    l1 = builder.aiGraphcore.l1loss([out], 0.1)

    anchorIds = []

    builder.addOutputTensor(out)

    device = tu.create_test_device(
        num_replicas * (2 if enable_pingpong else 1),
        pattern=popart.SyncPattern.Full)

    dfAnchors = {}
    for anchorId in anchorIds:
        dfAnchors.update({anchorId: popart.AnchorReturnType("All")})

    opts = popart.SessionOptions()
    opts.enableOutlining = enable_outlining
    opts.enableReplicatedGraphs = True if num_replicas > 1 else False
    opts.replicatedGraphCount = num_replicas
    opts.replicatedWeightSharding = replicated_weight_sharding
    opts.replicatedWeightShardingMinNumElements = 8

    if activation_tensor_location_settings is not None:
        opts.activationTensorLocationSettings = activation_tensor_location_settings
    if weight_tensor_location_settings is not None:
        opts.weightTensorLocationSettings = weight_tensor_location_settings
    if optimizer_state_tensor_location_settings is not None:
        opts.optimizerStateTensorLocationSettings = optimizer_state_tensor_location_settings

    opts.tensorLocationSettingsOverride = tensor_location_setting_override

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

    ir = json.loads(session._serializeIr(popart.IrSerializationFormat.JSON))
    return ir


def check_ir(ir, check_onchip, check_offchip):
    """ We expect check_onchip / check_offchip to be lists of tensor ids that we need to assert were on-chip / off-chip, 
        respectively. Note that we check this by looking for the presence / absence of tensors 'RemoteArg___<id>' in the IR.
    """

    # Extract all tensors from IR.
    ir_tensors = set()
    for op in ir['maingraph']:
        for inp in op['inputs']:
            ir_tensors.add(inp['name'])
        for outp in op['outputs']:
            ir_tensors.add(outp['name'])

    off_chip_ir_tensors = [
        t for t in ir_tensors if t.startswith(popart.reservedRemoteArgPrefix())
    ]

    #print("All tensors: %s" % ir_tensors)
    #print("All off chip tensors: %s" % [t for t in ir_tensors if t.startswith(popart.reservedRemoteArgPrefix())])

    # Check all tensors we need to check.
    for tensor in check_onchip:
        off_chip_tensor_name = popart.reservedRemoteArgPrefix() + tensor
        assert off_chip_tensor_name not in off_chip_ir_tensors, "Expected %s to be OnChip but a tensor named %s was found in the IR, indicating it's OffChip" % (
            tensor, off_chip_tensor_name)

    for tensor in check_offchip:
        off_chip_tensor_name = popart.reservedRemoteArgPrefix() + tensor
        assert off_chip_tensor_name in off_chip_ir_tensors, "Expected %s to be OffChip but no tensor named %s was found in the IR, indicating it's OnChip" % (
            tensor, off_chip_tensor_name)


def test_weight_tensor_location_settings():
    # Check weight tensor location settings work.
    ir = get_ir(weight_tensor_location_settings=None)
    check_ir(ir, check_onchip=[], check_offchip=['W0', 'W1', 'W2'])

    ir = get_ir(weight_tensor_location_settings=popart.TensorLocationSettings(
        popart.TensorLocation.OffChip, 0))
    check_ir(ir, check_onchip=[], check_offchip=['W0', 'W1', 'W2'])

    ir = get_ir(weight_tensor_location_settings=popart.TensorLocationSettings(
        popart.TensorLocation.OnChip, 0))
    check_ir(ir, check_onchip=['W0', 'W1', 'W2'], check_offchip=[])


def test_weight_tensor_location_settings_plus_override():
    # Check weight tensor location settings work.
    ir = get_ir(
        weight_tensor_location_settings=popart.TensorLocationSettings(
            popart.TensorLocation.OffChip, 0),
        tensor_location_setting_override={'W2': popart.TensorLocation.OnChip})
    check_ir(ir, check_onchip=['W2'], check_offchip=['W0', 'W1'])

    ir = get_ir(
        weight_tensor_location_settings=popart.TensorLocationSettings(
            popart.TensorLocation.OnChip, 0),
        tensor_location_setting_override={'W1': popart.TensorLocation.OffChip})
    check_ir(ir, check_onchip=['W0', 'W2'], check_offchip=['W1'])


def test_activation_tensor_location_settings():
    # Check weight tensor location settings work.
    ir = get_ir(num_layers=5, activation_tensor_location_settings=None)
    check_ir(ir,
             check_onchip=[],
             check_offchip=['MatMul:0/1__t6', 'MatMul:0__t3'])

    ir = get_ir(
        num_layers=5,
        activation_tensor_location_settings=popart.TensorLocationSettings(
            popart.TensorLocation.OffChip, 0))
    check_ir(ir,
             check_onchip=[],
             check_offchip=['MatMul:0/1__t6', 'MatMul:0__t3'])

    ir = get_ir(
        num_layers=5,
        activation_tensor_location_settings=popart.TensorLocationSettings(
            popart.TensorLocation.OnChip, 0))
    check_ir(ir,
             check_onchip=['MatMul:0/1__t6', 'MatMul:0__t3'],
             check_offchip=[])


def test_activation_tensor_location_settings_plus_override():
    # Check weight tensor location settings work.
    ir = get_ir(
        num_layers=5,
        activation_tensor_location_settings=popart.TensorLocationSettings(
            popart.TensorLocation.OffChip, 0),
        tensor_location_setting_override={
            'MatMul:0/1__t6': popart.TensorLocation.OnChip
        })
    check_ir(ir,
             check_onchip=['MatMul:0/1__t6'],
             check_offchip=['MatMul:0__t3'])

    ir = get_ir(
        num_layers=5,
        activation_tensor_location_settings=popart.TensorLocationSettings(
            popart.TensorLocation.OnChip, 0),
        tensor_location_setting_override={
            'MatMul:0/1__t6': popart.TensorLocation.OffChip
        })
    check_ir(ir,
             check_onchip=['MatMul:0__t3'],
             check_offchip=['MatMul:0/1__t6'])


def test_optimizer_state_tensor_location_settings():
    # Check optimizer state tensor location settings work.
    optimizer_with_state = popart.SGD({
        "defaultLearningRate": (0.1, True),
        "defaultMomentum": (0.0, False),
        "defaultWeightDecay": (0.0, False),
        "defaultDampening": (0.0, True)
    })
    ir = get_ir(optimizer_state_tensor_location_settings=None,
                optimizer=optimizer_with_state)
    check_ir(ir,
             check_onchip=[],
             check_offchip=['Accl___W1', 'Accl___W2', 'Accl___W0'])

    ir = get_ir(
        optimizer_state_tensor_location_settings=popart.TensorLocationSettings(
            popart.TensorLocation.OffChip, 0),
        optimizer=optimizer_with_state)
    check_ir(ir,
             check_onchip=[],
             check_offchip=['Accl___W1', 'Accl___W2', 'Accl___W0'])

    ir = get_ir(
        optimizer_state_tensor_location_settings=popart.TensorLocationSettings(
            popart.TensorLocation.OnChip, 0),
        optimizer=optimizer_with_state)
    check_ir(ir,
             check_onchip=['Accl___W1', 'Accl___W2', 'Accl___W0'],
             check_offchip=[])


def test_optimizer_state_tensor_location_settings_plus_override():
    # Check optimizer state tensor location settings work
    optimizer_with_state = popart.SGD({
        "defaultLearningRate": (0.1, True),
        "defaultMomentum": (0.0, False),
        "defaultWeightDecay": (0.0, False),
        "defaultDampening": (0.0, True)
    })
    ir = get_ir(
        optimizer_state_tensor_location_settings=popart.TensorLocationSettings(
            popart.TensorLocation.OffChip, 0),
        tensor_location_setting_override={
            'Accl___W1': popart.TensorLocation.OnChip
        },
        optimizer=optimizer_with_state)
    check_ir(ir,
             check_onchip=['Accl___W1'],
             check_offchip=['Accl___W2', 'Accl___W0'])

    ir = get_ir(
        optimizer_state_tensor_location_settings=popart.TensorLocationSettings(
            popart.TensorLocation.OnChip, 0),
        tensor_location_setting_override={
            'Accl___W1': popart.TensorLocation.OffChip
        },
        optimizer=optimizer_with_state)
    check_ir(ir,
             check_onchip=['Accl___W2', 'Accl___W0'],
             check_offchip=['Accl___W1'])
