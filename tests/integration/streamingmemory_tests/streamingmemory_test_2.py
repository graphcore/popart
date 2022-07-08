# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import popart
import onnx

# `import test_util` requires adding to sys.path
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu

from streamingmemory_test_0 import (
    run_model,
    check_model,
    onChipLocation,
    offChipLocation,
    offChipRtsLocation,
    onChipRtsLocation,
)


@tu.requires_ipu
def test_pipelined_streaming_rmsprop(tmpdir):

    optimizer_dict = {
        "defaultLearningRate": (0.005, False),
        "defaultMomentum": (0.8, False),
        "defaultWeightDecay": (0.1, False),
        "defaultEps": (1e-6, False),
        "lossScaling": (10.0, False),
    }

    optimizer = popart.Adaptive(
        optimizer_dict,
        popart.AdaptiveMode.RMSProp,
        popart.WeightDecayMode.L2Regularization,
        popart.DataType.FLOAT,
        popart.DataType.FLOAT,
        popart.DataType.FLOAT,
        popart.DataType.FLOAT,
        True,
    )

    run_model(
        tmpdir,
        "normal.onnx",
        execution_mode="normal",
        num_layers=2,
        batch_size=12,
        num_replicas=1,
        num_iterations=5,
        enable_accum=False,
        accum_factor=1,
        optimizer=optimizer,
        activation_tensor_location_settings=onChipLocation,
        weight_tensor_location_settings=onChipLocation,
        optimizer_state_tensor_location_settings=onChipLocation,
        accumulator_tensor_location_settings=onChipLocation,
    )
    run_model(
        tmpdir,
        "pipelined.onnx",
        execution_mode="pipelined",
        num_layers=2,
        batch_size=2,
        num_replicas=1,
        num_iterations=5,
        enable_accum=True,
        accum_factor=6,
        optimizer=optimizer,
        activation_tensor_location_settings=onChipLocation,
        weight_tensor_location_settings=onChipLocation,
        optimizer_state_tensor_location_settings=onChipLocation,
        accumulator_tensor_location_settings=onChipLocation,
    )
    run_model(
        tmpdir,
        "pipelined_streaming.onnx",
        execution_mode="pipelined",
        num_layers=2,
        batch_size=2,
        num_replicas=1,
        num_iterations=5,
        enable_accum=True,
        accum_factor=6,
        optimizer=optimizer,
        activation_tensor_location_settings=onChipLocation,
        weight_tensor_location_settings=onChipLocation,
        optimizer_state_tensor_location_settings=offChipRtsLocation,
        accumulator_tensor_location_settings=onChipLocation,
    )
    run_model(
        tmpdir,
        "pipelined_streaming_rep.onnx",
        execution_mode="pipelined",
        num_layers=2,
        batch_size=1,
        num_replicas=2,
        num_iterations=5,
        enable_accum=True,
        accum_factor=6,
        optimizer=optimizer,
        activation_tensor_location_settings=onChipLocation,
        weight_tensor_location_settings=onChipLocation,
        optimizer_state_tensor_location_settings=offChipLocation,
        accumulator_tensor_location_settings=onChipLocation,
    )
    run_model(
        tmpdir,
        "pipelined_streaming_rep_rts.onnx",
        execution_mode="pipelined",
        num_layers=2,
        batch_size=1,
        num_replicas=2,
        num_iterations=5,
        enable_accum=True,
        accum_factor=6,
        optimizer=optimizer,
        activation_tensor_location_settings=onChipLocation,
        weight_tensor_location_settings=onChipLocation,
        optimizer_state_tensor_location_settings=offChipRtsLocation,
        accumulator_tensor_location_settings=onChipLocation,
    )
    run_model(
        tmpdir,
        "pipelined_streaming_rep_rts_onchip.onnx",
        execution_mode="pipelined",
        num_layers=2,
        batch_size=1,
        num_replicas=2,
        num_iterations=5,
        enable_accum=True,
        accum_factor=6,
        optimizer=optimizer,
        activation_tensor_location_settings=onChipLocation,
        weight_tensor_location_settings=onChipLocation,
        optimizer_state_tensor_location_settings=onChipRtsLocation,
        accumulator_tensor_location_settings=onChipLocation,
    )

    normal = onnx.load(str(tmpdir / "normal.onnx"))
    pipelined = onnx.load(str(tmpdir / "pipelined.onnx"))
    pipelined_streaming = onnx.load(str(tmpdir / "pipelined_streaming.onnx"))
    pipelined_streaming_rep = onnx.load(str(tmpdir / "pipelined_streaming_rep.onnx"))
    pipelined_streaming_rep_rts = onnx.load(
        str(tmpdir / "pipelined_streaming_rep_rts.onnx")
    )
    pipelined_streaming_rep_rts_onchip = onnx.load(
        str(tmpdir / "pipelined_streaming_rep_rts_onchip.onnx")
    )

    check_model(normal, pipelined)
    check_model(normal, pipelined_streaming)
    check_model(normal, pipelined_streaming_rep)
    check_model(normal, pipelined_streaming_rep_rts)
    check_model(normal, pipelined_streaming_rep_rts_onchip)
