# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import popart
import onnx

# `import test_util` requires adding to sys.path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu

from streamingmemory_test_0 import run_model, check_model, onChipLocation, offChipLocation, offChipRtsLocation


@tu.requires_ipu
def test_sharding_without_replicas_warning(tmpdir):
    run_model(tmpdir,
              'warning.onnx',
              execution_mode="phased",
              num_replicas=1,
              activation_tensor_location_settings=offChipLocation,
              weight_tensor_location_settings=offChipRtsLocation,
              optimizer_state_tensor_location_settings=offChipRtsLocation,
              accumulator_tensor_location_settings=offChipRtsLocation)


@tu.requires_ipu
def test_inplacing_phased_constraints(tmpdir):
    # This used to fail, see T23985
    run_model(tmpdir,
              'phased.onnx',
              execution_mode="phased",
              num_layers=5,
              optimizer=popart.SGD({
                  "defaultLearningRate": (0.1, True),
                  "defaultMomentum": (0.0, False),
                  "defaultWeightDecay": (0.0, False),
                  "defaultDampening": (0.0, True)
              }),
              activation_tensor_location_settings=offChipLocation,
              weight_tensor_location_settings=offChipLocation,
              optimizer_state_tensor_location_settings=offChipLocation,
              accumulator_tensor_location_settings=offChipLocation)


@tu.requires_ipu
def test_pipelined_streaming_lamb(tmpdir):

    optimizer_dict = {
        "defaultLearningRate": (0.005, True),
        "defaultBeta1": (0.7, True),
        "defaultBeta2": (0.8, True),
        "defaultWeightDecay": (0.1, True),
        "defaultEps": (1e-6, True),
        "lossScaling": (10.0, True),
    }

    run_model(tmpdir,
              'normal.onnx',
              execution_mode="normal",
              num_layers=2,
              batch_size=12,
              num_replicas=1,
              num_iterations=5,
              enable_accum=False,
              accum_factor=1,
              optimizer=popart.Adam(optimizer_dict, popart.AdamMode.Lamb),
              activation_tensor_location_settings=onChipLocation,
              weight_tensor_location_settings=onChipLocation,
              optimizer_state_tensor_location_settings=onChipLocation,
              accumulator_tensor_location_settings=onChipLocation)
    run_model(tmpdir,
              'pipelined.onnx',
              execution_mode="pipelined",
              num_layers=2,
              batch_size=2,
              num_replicas=1,
              num_iterations=5,
              enable_accum=True,
              accum_factor=6,
              optimizer=popart.Adam(optimizer_dict, popart.AdamMode.Lamb),
              activation_tensor_location_settings=onChipLocation,
              weight_tensor_location_settings=onChipLocation,
              optimizer_state_tensor_location_settings=onChipLocation,
              accumulator_tensor_location_settings=onChipLocation)
    run_model(tmpdir,
              'pipelined_streaming.onnx',
              execution_mode="pipelined",
              num_layers=2,
              batch_size=2,
              num_replicas=1,
              num_iterations=5,
              enable_accum=True,
              accum_factor=6,
              optimizer=popart.Adam(optimizer_dict, popart.AdamMode.Lamb),
              activation_tensor_location_settings=onChipLocation,
              weight_tensor_location_settings=onChipLocation,
              optimizer_state_tensor_location_settings=offChipRtsLocation,
              accumulator_tensor_location_settings=onChipLocation)
    run_model(tmpdir,
              'pipelined_streaming_rep.onnx',
              execution_mode="pipelined",
              num_layers=2,
              batch_size=1,
              num_replicas=2,
              num_iterations=5,
              enable_accum=True,
              accum_factor=6,
              optimizer=popart.Adam(optimizer_dict, popart.AdamMode.Lamb),
              activation_tensor_location_settings=onChipLocation,
              weight_tensor_location_settings=onChipLocation,
              optimizer_state_tensor_location_settings=offChipLocation,
              accumulator_tensor_location_settings=onChipLocation)
    run_model(tmpdir,
              'pipelined_streaming_rep_rts.onnx',
              execution_mode="pipelined",
              num_layers=2,
              batch_size=1,
              num_replicas=2,
              num_iterations=5,
              enable_accum=True,
              accum_factor=6,
              optimizer=popart.Adam(optimizer_dict, popart.AdamMode.Lamb),
              activation_tensor_location_settings=onChipLocation,
              weight_tensor_location_settings=onChipLocation,
              optimizer_state_tensor_location_settings=offChipRtsLocation,
              accumulator_tensor_location_settings=onChipLocation)

    normal = onnx.load(str(tmpdir / 'normal.onnx'))
    pipelined = onnx.load(str(tmpdir / 'pipelined.onnx'))
    pipelined_streaming = onnx.load(str(tmpdir / 'pipelined_streaming.onnx'))
    pipelined_streaming_rep = onnx.load(
        str(tmpdir / 'pipelined_streaming_rep.onnx'))
    pipelined_streaming_rep_rts = onnx.load(
        str(tmpdir / 'pipelined_streaming_rep_rts.onnx'))

    check_model(normal, pipelined)
    check_model(normal, pipelined_streaming)
    check_model(normal, pipelined_streaming_rep)
    check_model(normal, pipelined_streaming_rep_rts)
