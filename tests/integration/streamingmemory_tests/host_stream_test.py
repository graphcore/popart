# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
'''
These test compare the output of some models with bufferStreamCopiesToDevice on vs off.
With various other common settings.
'''

import numpy as np
import popart
import json
import pytest
from typing import Dict, List, Tuple

# `import test_util` requires adding to sys.path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu

BPS = 12
LR = 1e-1
np.random.seed(42)

# Generate some random input data
trainingData = np.random.rand(BPS, 8, 2).astype(np.float16)
trainingDataLables = np.random.rand(BPS, 8).astype(np.int32)
w_init = np.random.rand(2, 2).astype(np.float16)
bias_init = np.random.rand(2).astype(np.float16)


def create_model_pipelined(bufferStreams: bool = False,
                           pipelining: bool = False,
                           device=None) -> Dict:
    """Create a simple model with optional pipeliing to test buffering streams

    Args:
        bufferStreams (bool, optional): Whether bufferStreamCopiesToDevice is on or off.
            Defaults to False.
        pipelining (bool, optional): Whether to pipeline the model in 2 parts.
            Defaults to False.
        device (Optional[DeviceContext]): The device.

    Returns:
        Dict: A dict of session, stepio, anchors and out tensor name required
            to run and test the model.
    """
    builder = popart.Builder()

    data_shape = popart.TensorInfo("FLOAT16", [8, 2])
    lbl_shape = popart.TensorInfo("INT32", [8])

    ip = builder.addInputTensor(data_shape, "input_data")
    lb = builder.addInputTensor(lbl_shape, "label")

    w = builder.addInitializedInputTensor(w_init)
    b = builder.addInitializedInputTensor(bias_init)
    gemm = builder.aiOnnx.gemm([ip, w, b], 1., 1., False, False)
    relu = builder.aiOnnx.relu([gemm])
    sm = builder.aiOnnx.softmax([relu])
    nll = builder.aiGraphcore.nllloss([sm, lb])

    builder.addOutputTensor(sm)

    art = popart.AnchorReturnType("All")
    dataFlow = popart.DataFlow(BPS, {sm: art, nll: art})

    opts = popart.SessionOptions()
    opts.enableOutlining = True
    opts.useHostCopyOps = bufferStreams

    if pipelining:
        opts.enablePipelining = True
        opts.virtualGraphMode = popart.VirtualGraphMode.Manual
        builder.pipelineStage(gemm, 0)
        builder.virtualGraph(gemm, 0)
        builder.pipelineStage(relu, 0)
        builder.virtualGraph(relu, 0)
        builder.pipelineStage(sm, 1)
        builder.virtualGraph(sm, 1)
        builder.pipelineStage(nll, 1)
        builder.virtualGraph(nll, 1)

    session = popart.TrainingSession(fnModel=builder.getModelProto(),
                                     dataFlow=dataFlow,
                                     loss=nll,
                                     optimizer=popart.ConstSGD(0.1),
                                     userOptions=opts,
                                     deviceInfo=device)

    session.prepareDevice()

    # 2 host load ops for input
    check_ops(session, bufferStreams, 2)

    anchors = session.initAnchorArrays()
    stepio = popart.PyStepIO({
        ip: trainingData,
        lb: trainingDataLables
    }, anchors)

    return {
        "session": session,
        "stepio": stepio,
        "anchors": anchors,
        "out": sm
    }


def check_ops(session: popart.TrainingSession, bufferStreams: bool,
              numOps: int) -> None:
    """Check the number of host load and int ops are as expected

    Args:
        session (popart.TrainingSession): The session to check
        bufferStreams (bool): Whether bufferStreamCopiesToDevice is on or off
        numOps (int): The number of expected ops (== to the number of data input streams,
            excluding random seed streams, anchor streams etc.)
    """
    ir = json.loads(session._serializeIr(popart.IrSerializationFormat.JSON))

    ops = ir["maingraph"]

    initOps = [op for op in ops if op["type"] == "Init"]
    hostLoadOps = [op for op in ops if op["type"] == "HostLoad"]

    if bufferStreams:
        assert len(initOps) == numOps
        assert len(hostLoadOps) == numOps
    else:
        assert len(initOps) == 0
        assert len(hostLoadOps) == 0


# Pipelining disabled until explicit pipelining supported (T35924)
@tu.requires_ipu_model
@pytest.mark.parametrize("enable_pipelining", [False])
def test_basic_host_load_output(enable_pipelining: bool):
    """Run a simple model comparing outputs with buffer stream copies on and off.
    See T29603 for description of bufferStreamCopiesToDevice

    Args:
        enable_pipelining (bool): Whether to split the model in two and pipeline it.
        Parameterized on and off.
    """
    numIPUs = 1
    if enable_pipelining:
        numIPUs = 2

    with tu.create_test_device(numIPUs) as d1, tu.create_test_device(
            numIPUs) as d2:
        bundle_false = create_model_pipelined(bufferStreams=False,
                                              pipelining=enable_pipelining,
                                              device=d1)
        bundle_true = create_model_pipelined(bufferStreams=True,
                                             pipelining=enable_pipelining,
                                             device=d2)

        for step in range(5):
            print(f"Running step {step}")
            for bundle in (bundle_false, bundle_true):
                bundle["session"].weightsFromHost()
                bundle["session"].run(bundle["stepio"])
                bundle["session"].weightsToHost()
            print("\tChecking", bundle_false["out"], "vs", bundle_true["out"])
            assert np.allclose(bundle_false["anchors"][bundle_false["out"]],
                               bundle_true["anchors"][bundle_true["out"]])


def get_model(input_shape: List[int], weight_array: np.array,
              batches_per_step: int, replication_factor: int, batch_size: int,
              channels: int, data_len: int, synthetic_data: bool,
              buffer_streams: bool, device) -> Tuple:
    """Get a simple model for comparison with buffer streams on and off.
    Adapted from prefetch_test.py as we require to test the validity of streams
    here as well.

    Args:
        input_shape (List[int]): The input shapes of the model
        weight_array (np.array): The weights.
        batches_per_step (int): Batches to run per step.
        replication_factor (int): Replicas to run.
        batch_size (int): Number of samples per model run.
        channels (int): Number of channels e.g. RGB = 3.
        data_len (int): Data size.
        synthetic_data (bool): Use synthetic data (zeros in this case).
        buffer_streams (bool): The test option: whether to create ops
            before the stream in order to schedule data loading as part of
            graph scheduling. See T29603.
        device (DeviceContext): The device.

    Returns:
        Tuple: session, anchors, input_shape, label_shape required to run the model
    """

    micro_batch_size = batch_size // (replication_factor)

    builder = popart.Builder()

    data_shape = popart.TensorInfo("FLOAT", input_shape)
    lbl_shape = popart.TensorInfo("INT32", [micro_batch_size])
    w = builder.addInitializedInputTensor(weight_array)

    ip = builder.addInputTensor(data_shape, "main_input_123")
    lb = builder.addInputTensor(lbl_shape, "label_input_456")

    a = builder.aiOnnx.matmul([ip, w])
    o = builder.reshape_const(
        builder.aiOnnx, [a],
        [micro_batch_size, channels * data_len * data_len])
    relu = builder.aiOnnx.relu([o])
    sm = builder.aiOnnx.softmax([relu], axis=0, debugContext="output")
    builder.addOutputTensor(sm)
    o = builder.aiGraphcore.nllloss([sm, lb],
                                    reduction=popart.ReductionType.Mean)

    art = popart.AnchorReturnType("All")
    data_flow = popart.DataFlow(batches_per_step, {
        ip: art,
        lb: art,
        o: art,
        sm: art,
        a: art,
        relu: art
    })

    opts = popart.SessionOptions()
    opts.useHostCopyOps = buffer_streams
    # TODO: Fix outlining
    opts.enableOutlining = False

    ipus = 1

    if replication_factor > 1:
        opts.replicatedGraphCount = replication_factor
        opts.enableReplicatedGraphs = True
        ipus *= replication_factor

    assert device

    patterns = popart.Patterns(popart.PatternsLevel.Minimal).enablePattern(
        "MatMulLhsGradOp", True).enablePattern("MatMulRhsGradOp", True)
    patterns.InPlace = False
    if synthetic_data:
        opts.syntheticDataMode = popart.SyntheticDataMode.Zeros

    session = popart.TrainingSession(fnModel=builder.getModelProto(),
                                     dataFlow=data_flow,
                                     loss=o,
                                     optimizer=popart.ConstSGD(LR),
                                     userOptions=opts,
                                     deviceInfo=device,
                                     patterns=patterns)

    session.setRandomSeed(0)
    session.prepareDevice()

    label_shape = [micro_batch_size]

    if replication_factor > 1:
        input_shape = [replication_factor] + input_shape
        label_shape = [replication_factor] + label_shape
    if batches_per_step > 1:
        input_shape = [batches_per_step] + input_shape
        label_shape = [batches_per_step] + label_shape

    anchors = session.initAnchorArrays()

    return session, anchors, label_shape


def run_model(session: popart.TrainingSession, anchors: Dict,
              in_array: np.array,
              label_array: np.array) -> Tuple[np.array, np.array, np.array]:
    """Run the model using the provided params

    Args:
        session (popart.TrainingSession): The compiled session to use
        anchors (Dict): The anchors to test
        in_array (np.array): The input array
        label_array (np.array): The label array

    Returns:
        Tuple[np.array, np.array, np.array]: The results from the run.
    """
    stepio = popart.PyStepIO(
        {
            "main_input_123": in_array,
            "label_input_456": label_array
        }, anchors)
    session.weightsFromHost()

    session.run(stepio)

    session.weightsToHost()

    return anchors["main_input_123"], anchors["Relu:0"], anchors[
        "Nll:0"], anchors["MatMul:0"]


def run_test(batches_per_step: int, replication_factor: int, batch_size: int,
             channels: int, data_len: int, steps: int) -> None:
    """Run the test for the provided steps and check outputs.

    Args:
        batches_per_step (int): Batches to run per step
        replication_factor (int): Replicas to run
        batch_size (int): Number of samples per model run
        channels (int): Number of channels e.g. RGB = 3
        data_len (int): Data size
        steps (int): Number of steps to run
    """

    micro_batch_size = batch_size // replication_factor
    input_shape = [micro_batch_size, channels, data_len, data_len]
    weight_array = np.random.random_sample(input_shape).astype(np.float32)

    ipus = 1
    if replication_factor > 1:
        ipus *= replication_factor

    with tu.create_test_device(ipus) as d1, tu.create_test_device(ipus) as d2:
        sesstion_true, anchors_true, label_shape_true = get_model(
            input_shape=input_shape,
            weight_array=weight_array,
            batches_per_step=batches_per_step,
            replication_factor=replication_factor,
            batch_size=batch_size,
            channels=channels,
            data_len=data_len,
            synthetic_data=False,
            buffer_streams=True,  # <-- Testing this
            device=d1)

        # check_ops(sesstion_true, True, 2)

        session_false, anchors_false, label_shape_false = get_model(
            input_shape=input_shape,
            weight_array=weight_array,
            batches_per_step=batches_per_step,
            replication_factor=replication_factor,
            batch_size=batch_size,
            channels=channels,
            data_len=data_len,
            synthetic_data=False,
            buffer_streams=False,  # <-- Testing this
            device=d2)

        for step in range(steps):
            print("Step:", step + 1)
            in_array = (np.random.random_sample([
                batch_size * batches_per_step, channels, data_len, data_len
            ]).astype(np.float32) + 1) * 10

            label_array = np.random.randint(
                low=0,
                high=(channels * data_len * data_len),
                size=label_shape_true).astype(np.int32)
            # Only provide one session.run's worth of data.
            tuple_1 = run_model(sesstion_true, anchors_true, in_array,
                                label_array)

            tuple_2 = run_model(session_false, anchors_false, in_array,
                                label_array)

            for i in range(batches_per_step):
                if batches_per_step == 1:
                    i = None
                else:
                    print("  Batch:", i)

                for anchor_1, anchor_2 in zip(tuple_1, tuple_2):
                    print("Host IO ops:", np.mean(anchor_1[i]))
                    print("No Host IO ops:", np.mean(anchor_2[i]))
                    assert np.allclose(anchor_1[i], anchor_2[i])


@tu.requires_ipu_model
def test_host_load_simple():
    """Run a simple model
    """
    args = dict(batches_per_step=1,
                replication_factor=1,
                batch_size=1,
                channels=3,
                data_len=4,
                steps=2)
    run_test(**args)


@tu.requires_ipu_model
def test_host_load_multi_bps():
    """Run a simple model with bps > 1
    """
    args = dict(batches_per_step=5,
                replication_factor=1,
                batch_size=4,
                channels=3,
                data_len=4,
                steps=2)
    run_test(**args)


@tu.requires_ipu
@pytest.mark.skip("Not working with replication yet")
def test_host_load_replicated():
    """Run a replicated model
    """
    args = dict(batches_per_step=7,
                replication_factor=2,
                batch_size=4,
                channels=3,
                data_len=2,
                steps=1)
    run_test(**args)


@tu.requires_ipu_model
def test_host_load_pipeline():
    """Run with pipelining
    """
    args = dict(batches_per_step=7,
                replication_factor=1,
                batch_size=1,
                channels=2,
                data_len=3,
                steps=1)
    run_test(**args)
