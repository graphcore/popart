# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
import json

# 'import test_util' requires adding to sys.path
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import test_util as tu


@tu.requires_ipu_model
def test_full_recompute_pipelining():
    np.random.seed(0)

    gradient_accumulation = 5
    batch_size = 1
    hidden_size = 16

    input_shape = [batch_size, hidden_size]

    weight_data = np.random.normal(0, 0.02, [hidden_size, hidden_size]).astype(
        np.float32
    )

    input_data = np.random.normal(
        0, 0.02, [gradient_accumulation] + input_shape
    ).astype(np.float32)

    def run_test(mode=None, verify=None):
        builder = popart.Builder()

        def norm(input_x):
            gamma = builder.addInitializedInputTensor(
                np.ones(hidden_size, np.float32), "Gamma"
            )
            beta = builder.addInitializedInputTensor(
                np.zeros(hidden_size, np.float32), "Beta"
            )
            return builder.aiGraphcore.groupnormalization([input_x, gamma, beta], 1)[0]

        x_in = builder.addInputTensor(popart.TensorInfo("FLOAT", input_shape), "x_in")

        weight_1 = builder.addInitializedInputTensor(weight_data, "weight_1")
        weight_2 = builder.addInitializedInputTensor(weight_data, "weight_2")
        weight_3 = builder.addInitializedInputTensor(weight_data, "weight_3")

        with builder.virtualGraph(0), builder.pipelineStage(0):
            x_0 = builder.aiOnnx.matmul([x_in, weight_1])
            x_0 = norm(x_0)

            # If recomputeOutputs was used directly on `x_0` all 3 outputs
            # of groupnormalization would be stashed.
            # By using a checkpointOutput only 1 output will be stashed and the
            # rest will be recomputed.
            x_0 = builder.checkpointOutput([x_0])[0]

            x_1 = builder.aiOnnx.matmul([x_0, weight_2])
            x_1 = norm(x_1)
            x_1 = builder.aiOnnx.add([x_0, x_1])

            # This checkpoint should be redundant as x_1 will be stashed
            # at the start of stage1 on ipu1.
            x_1 = builder.checkpointOutput([x_1])[0]

        with builder.virtualGraph(1), builder.pipelineStage(1):
            o = builder.aiOnnx.matmul([x_1, weight_3])
            l1 = builder.aiGraphcore.l1loss([o], 0.1)

        proto = builder.getModelProto()

        dataFlow = popart.DataFlow(
            1,
            [
                o,
                popart.reservedGradientPrefix() + weight_1,
                popart.reservedGradientPrefix() + weight_2,
                popart.reservedGradientPrefix() + weight_3,
            ],
        )

        opts = popart.SessionOptions()
        opts.enableOutlining = False
        opts.enablePipelining = True
        opts.enableGradientAccumulation = True
        opts.accumulationFactor = gradient_accumulation
        opts.optimizerStateTensorLocationSettings.location.storage = (
            popart.TensorStorage.OffChip
        )
        if mode is not None:
            opts.autoRecomputation = mode
        opts.virtualGraphMode = popart.VirtualGraphMode.Manual

        with tu.create_test_device(numIpus=2, opts={"compileIPUCode": False}) as device:
            session = popart.TrainingSession(
                fnModel=proto,
                dataFlow=dataFlow,
                userOptions=opts,
                loss=l1,
                optimizer=popart.Adam({}),
                deviceInfo=device,
            )

            session.prepareDevice()

            session.weightsFromHost()

            anchors = session.initAnchorArrays()

            inputs = {x_in: input_data}
            stepio = popart.PyStepIO(inputs, anchors)

            for _ in range(10):
                session.run(stepio)

            if verify is not None:
                verify(session, x_0)

            return anchors

    def verify(session, mid_stash):
        """Verify the the matmul in the main graphs is correct"""
        ir = json.loads(session._serializeIr(popart.IrSerializationFormat.JSON))
        stashes = [op for op in ir["maingraph"] if op["type"] == "Stash"]
        stashedTensors = [stash["inputs"][0]["name"] for stash in stashes]

        assert {"x_in", mid_stash} == set(stashedTensors)

    n_anchors = run_test()
    p_anchors = run_test(popart.RecomputationType.Pipeline, verify)

    for key in n_anchors:
        assert np.allclose(n_anchors[key], p_anchors[key])


@tu.requires_ipu_model
def test_delayed_restore_operations():
    np.random.seed(0)

    gradient_accumulation = 5
    batch_size = 1
    hidden_size = 16

    input_shape = [batch_size, hidden_size]

    weight_data = np.random.normal(0, 0.02, [hidden_size, hidden_size]).astype(
        np.float32
    )

    input_data = np.random.normal(
        0, 0.02, [gradient_accumulation] + input_shape
    ).astype(np.float32)

    def run_test(mode=None, verify=None):
        builder = popart.Builder()

        x_in = builder.addInputTensor(popart.TensorInfo("FLOAT", input_shape), "x_in")

        weight_1 = builder.addInitializedInputTensor(weight_data, "weight_1")

        # We want a bwd pass that looks like:
        #
        # restore, op1, restore, op2, restore, op3
        #
        # Where op1, op2 & op3 are gradient operations that
        # have implicit recompute inputs.

        with builder.virtualGraph(0), builder.pipelineStage(0):
            x = builder.aiOnnx.matmul([x_in, weight_1])
            x = builder.checkpointOutput([x])[0]

            x = builder.aiOnnx.add([x, x])
            # Gelu is a unary operation that takes the fwd input
            # activation. This satisfies our requirement above
            # of needing an implicit recompute input.
            x = builder.aiGraphcore.gelu([x])

            x = builder.checkpointOutput([x])[0]

            x = builder.aiOnnx.add([x, x])
            x = builder.aiGraphcore.gelu([x])

            x = builder.checkpointOutput([x])[0]
            o = x

        with builder.virtualGraph(1), builder.pipelineStage(1):
            l1 = builder.aiGraphcore.l1loss([o], 0.1)

        proto = builder.getModelProto()

        dataFlow = popart.DataFlow(
            1,
            [
                o,
                popart.reservedGradientPrefix() + weight_1,
            ],
        )

        opts = popart.SessionOptions()
        opts.enableOutlining = False
        opts.enablePipelining = True
        opts.enableGradientAccumulation = True
        opts.accumulationFactor = gradient_accumulation
        opts.optimizerStateTensorLocationSettings.location.storage = (
            popart.TensorStorage.OffChip
        )
        if mode is not None:
            opts.autoRecomputation = mode
        opts.virtualGraphMode = popart.VirtualGraphMode.Manual

        with tu.create_test_device(numIpus=2, opts={"compileIPUCode": False}) as device:
            session = popart.TrainingSession(
                fnModel=proto,
                dataFlow=dataFlow,
                userOptions=opts,
                loss=l1,
                optimizer=popart.Adam({}),
                deviceInfo=device,
            )

            session.prepareDevice()

            session.weightsFromHost()

            anchors = session.initAnchorArrays()

            inputs = {x_in: input_data}
            stepio = popart.PyStepIO(inputs, anchors)

            for _ in range(10):
                session.run(stepio)

            if verify is not None:
                verify(session)

            return anchors

    def verify(session):
        """Verify the the matmul in the main graphs is correct"""
        ir = json.loads(session._serializeIr(popart.IrSerializationFormat.JSON))
        schedule_string = ""
        for op in ir["maingraph"]:
            if "Add" in op["type"]:
                schedule_string += "a"
            elif "GeluGrad" in op["type"]:
                schedule_string += "g"
            elif "RestoreInplace" in op["type"]:
                schedule_string += "r"
            else:
                schedule_string += "_"
        assert schedule_string.count("rga") == 2

    n_anchors = run_test()
    p_anchors = run_test(popart.RecomputationType.Pipeline, verify)

    for key in n_anchors:
        assert np.allclose(n_anchors[key], p_anchors[key])


@tu.requires_ipu_model
def test_final_stage_recompute_0():
    np.random.seed(0)

    gradient_accumulation = 5
    batch_size = 1
    hidden_size = 16

    input_shape = [batch_size, hidden_size]
    weight_data = np.random.normal(0, 0.02, [hidden_size, hidden_size]).astype(
        np.float32
    )

    builder = popart.Builder()

    x_in = builder.addInputTensor(popart.TensorInfo("FLOAT", input_shape), "x_in")

    with builder.virtualGraph(0), builder.pipelineStage(0):
        weight_1 = builder.addInitializedInputTensor(weight_data, "weight_1")
        x = builder.aiOnnx.matmul([x_in, weight_1])

    with builder.virtualGraph(1), builder.pipelineStage(1):
        weight_2 = builder.addInitializedInputTensor(weight_data, "weight_2")
        x_recomp = builder.aiOnnx.matmul([x, weight_2])
        # This MatMul should be recomputed
        x = builder.checkpointOutput([x_recomp])[0]

        weight_3 = builder.addInitializedInputTensor(weight_data, "weight_3")
        # This MatMul should not be recomputed
        x_no_recomp = builder.aiOnnx.matmul([x, weight_3])
        l1 = builder.aiGraphcore.l1loss([x_no_recomp], 0.1)

    proto = builder.getModelProto()

    dataFlow = popart.DataFlow(1, [l1])

    opts = popart.SessionOptions()
    opts.enableOutlining = False
    opts.enablePipelining = True
    opts.enableGradientAccumulation = True
    opts.accumulationFactor = gradient_accumulation
    opts.optimizerStateTensorLocationSettings.location.storage = (
        popart.TensorStorage.OffChip
    )
    opts.autoRecomputation = popart.RecomputationType.Pipeline
    opts.virtualGraphMode = popart.VirtualGraphMode.Manual

    with tu.create_test_device(numIpus=2, opts={"compileIPUCode": False}) as device:
        session = popart.TrainingSession(
            fnModel=proto,
            dataFlow=dataFlow,
            userOptions=opts,
            loss=l1,
            optimizer=popart.Adam({}),
            deviceInfo=device,
        )
        """ Verify the the matmul in the main graphs is correct"""
        ir = json.loads(session._serializeIr(popart.IrSerializationFormat.JSON))

        for op in ir["maingraph"]:
            if x_recomp in map(lambda out: out["name"], op["outputs"]):
                assert op["attributes"]["recompute"] == "YES"
            elif x_no_recomp in map(lambda out: out["name"], op["outputs"]):
                assert op["attributes"]["recompute"] == "NO"


@tu.requires_ipu_model
def test_final_stage_recompute_1():
    """
    Model:
    out = NllLoss(Softmax(Matmul(x, w)), Cast(label))

    {Matmul} : ps0, {Nll, Softmax, Cast} : ps1

    Note that there are two 'paths' to the loss on the final forward
    PipelineStage: {Cast -> NllLoss} and {Softmax -> NllLoss}.

    Veryify that auto-recomputation in the final forward PipelineStage
    when doing RecomputationType::Pipeline auto-recomputation behaves
    as expected.

    If you checkpoint an operation on one path to the loss, do not expect to
    see ops on the other path recomputed.
    """

    builder = popart.Builder()
    x = builder.addInputTensor("FLOAT16", [2, 4])
    w_data = np.random.rand(4, 3).astype(np.float16)
    w = builder.addInitializedInputTensor(w_data)
    label = builder.addInputTensor("INT32", [2])

    with builder.virtualGraph(0), builder.pipelineStage(0):
        mm = builder.aiOnnx.matmul([x, w])

    with builder.virtualGraph(1), builder.pipelineStage(1):
        sfm = builder.aiOnnx.softmax([mm])
        label = builder.aiOnnx.cast([label], "UINT32")
        loss = builder.aiGraphcore.nllloss([sfm, label])

    # Checkpoint tensor along one path to the loss
    builder.recomputeOutputInBackwardPass(label, popart.RecomputeType.Checkpoint)

    opts = popart.SessionOptions()
    opts.autoRecomputation = popart.RecomputationType.Pipeline
    opts.enablePipelining = True
    opts.virtualGraphMode = popart.VirtualGraphMode.Manual

    with tu.create_test_device(numIpus=2, opts={"compileIPUCode": False}) as device:
        s = popart.TrainingSession(
            fnModel=builder.getModelProto(),
            userOptions=opts,
            loss=loss,
            optimizer=popart.ConstSGD(0.1),
            deviceInfo=device,
            dataFlow=popart.DataFlow(3, [loss]),
        )
        ir = json.loads(s._serializeIr(popart.IrSerializationFormat.JSON))

        recomputed_ops = []
        for op in ir["maingraph"]:
            finalFwdPipelineStage = "1"
            if op["attributes"]["__pipeline_stage"] == finalFwdPipelineStage:
                if op["attributes"]["recompute"] == "YES":
                    recomputed_ops.append(op)

        assert len(recomputed_ops) == 0
