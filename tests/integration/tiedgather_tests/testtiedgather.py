# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import json
import sys
from pathlib import Path

import numpy as np
import pytest

import popart

sys.path.append(str(Path(__file__).resolve().parent.parent))

from tiedgather_testutils import check_onnx_model, check_tensors, run_py


def model(splits=1, transpose_on_gather=True):
    np.random.seed(1984)
    input_data = np.random.randint(0, 20, (4,)).astype(np.uint32)
    weight_data = np.random.rand(4, 20).astype(np.float32)
    if not transpose_on_gather:
        weight_data = weight_data.T

    builder = popart.Builder()
    d0 = builder.addInputTensor(popart.TensorInfo("UINT32", input_data.shape), "data0")

    w0 = builder.addInitializedInputTensor(weight_data, "weight0")
    w0_t = builder.aiOnnx.transpose([w0], debugContext="weight0^T")

    x0 = builder.aiOnnx.gather(
        [w0_t if transpose_on_gather else w0, d0], debugContext="gathered"
    )

    x = builder.aiOnnx.matmul(
        [x0, w0 if transpose_on_gather else w0_t], debugContext="matmuld"
    )
    if splits > 1:
        builder.setSerializeMatMul({x}, "output_channels", splits, True)
    loss = builder.aiGraphcore.l1loss([x], 0.1, debugContext="loss")

    return builder.getModelProto(), {d0: input_data}, x, loss


def session(
    train=False,
    skip_execution=False,
    include_patterns=True,
    splits=1,
    outline=False,
    transpose_on_gather=True,
    optim="Sgd",
    mean_reduction_strategy=popart.MeanReductionStrategy.Running,
):
    proto, data, x, loss = model(splits=splits, transpose_on_gather=transpose_on_gather)

    patterns = popart.Patterns()
    patterns.enablePattern("TiedGather", include_patterns)
    patterns.enablePattern("TiedGatherAccumulate", include_patterns)

    user_options = {
        "enableOutlining": outline,
        "enableGradientAccumulation": True,
        "accumulationFactor": 2,
        "accumulationAndReplicationReductionType": popart.ReductionType.Mean,
        "meanAccumulationAndReplicationReductionStrategy": mean_reduction_strategy,
    }

    if optim == "Lamb":
        patterns.enablePattern("LambSerialisedWeight", True)

        optimizer = popart.Adam(
            {
                "defaultLearningRate": (0.1, False),
                "defaultWeightDecay": (0.1, True),
                "lossScaling": (20, True),
                "defaultBeta1": (0.1, True),
                "defaultBeta2": (0.1, True),
            },
            mode=popart.AdamMode.LambNoBias,
        )  # NoBias to increase the error of incorrect gradients
        user_options[
            "optimizerStateTensorLocationSettings"
        ] = popart.TensorLocationSettings(
            popart.TensorLocation(
                popart.TensorStorage.OffChip, popart.ReplicatedTensorSharding.On
            ),
            0,
            0,
        )
        user_options["enableReplicatedGraphs"] = True
        user_options["replicatedGraphCount"] = 2
    else:
        optimizer = popart.SGD(
            {
                "defaultLearningRate": (0.1, True),
                "defaultMomentum": (0.9, True),
                "defaultDampening": (
                    0,
                    True,
                ),  # 0 dampening to increase the error of incorrect gradients
                "lossScaling": (20, True),
            }
        )

    if train:
        return run_py(
            proto,
            data=data,
            outputs=x,
            loss=loss,
            optimizer=optimizer,
            patterns=patterns,
            user_options=user_options,
            skip_execution=skip_execution,
        )
    else:
        return run_py(
            proto,
            data=data,
            outputs=x,
            patterns=patterns,
            user_options={"enableOutlining": outline, "constantWeights": False},
            skip_execution=skip_execution,
        )


@pytest.mark.parametrize("splits", (1, 4))
@pytest.mark.parametrize("transpose_on_gather", (True, False))
@pytest.mark.parametrize(
    "mean_reduction_strategy",
    (popart.MeanReductionStrategy.Post, popart.MeanReductionStrategy.Running),
)
@pytest.mark.parametrize(
    ["phase", "optimizer"], [("fwd", None), ("bwd", "Sgd"), ("bwd", "Lamb")]
)
def test_tied_gather_pattern_ir(
    splits, phase, optimizer, transpose_on_gather, mean_reduction_strategy
):
    train = phase == "bwd"

    sess = session(
        train,
        skip_execution=True,
        splits=splits,
        optim=optimizer,
        transpose_on_gather=transpose_on_gather,
        outline=False,
        mean_reduction_strategy=mean_reduction_strategy,
    )

    ir = json.loads(sess._serializeIr(popart.IrSerializationFormat.JSON))

    ops = ir["maingraph"]

    # The gatherOp should be replaced with TiedGather
    assert len(list(filter(lambda op: op["type"] == "PopartTiedGather", ops))) == splits
    assert len(list(filter(lambda op: op["type"] == "Gather", ops))) == 0

    # The matmuls should have fully_connected_pass disabled
    assert all(
        map(
            lambda op: op["attributes"]["fully_connected_pass"] == "-1",
            filter(lambda op: op["type"] == "MatMul", ir["maingraph"]),
        )
    )

    if train:
        assert (
            len(list(filter(lambda op: op["type"] == "PopartSparseAccumulate", ops)))
            == splits
        )

        inv_op = list(
            filter(
                lambda op: op["name"] == "mean_accumulate_inverse"
                and op["type"] == "Div",
                ops,
            )
        )
        if (
            mean_reduction_strategy == popart.MeanReductionStrategy.Running
            and optimizer != "Sgd"
        ):
            assert len(inv_op) == 1
        else:
            assert len(inv_op) == 0


@pytest.mark.parametrize("splits", (1, 4))
@pytest.mark.parametrize("transpose_on_gather", (True, False))
@pytest.mark.parametrize(
    "mean_reduction_strategy",
    (popart.MeanReductionStrategy.Post, popart.MeanReductionStrategy.Running),
)
@pytest.mark.parametrize(
    ["phase", "optimizer"], [("fwd", None), ("bwd", "Sgd"), ("bwd", "Lamb")]
)
def test_tied_gather_pattern_correctness(
    splits, phase, optimizer, transpose_on_gather, mean_reduction_strategy
):
    train = phase == "bwd"

    outputs_1, proto_1, outnames_1 = session(
        train,
        skip_execution=False,
        splits=splits,
        optim=optimizer,
        transpose_on_gather=transpose_on_gather,
        outline=True,
        mean_reduction_strategy=mean_reduction_strategy,
    )

    outputs_2, proto_2, outnames_2 = session(
        train,
        skip_execution=False,
        include_patterns=False,
        splits=splits,
        optim=optimizer,
        transpose_on_gather=transpose_on_gather,
        outline=True,
        mean_reduction_strategy=mean_reduction_strategy,
    )

    check_tensors(outputs_1, outputs_2, outnames_1, outnames_2)
    if train:
        check_onnx_model(proto_1, proto_2)
