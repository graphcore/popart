# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import re

import numpy as np
import pytest

import popart
import test_util as tu
from unittest.mock import patch


@pytest.fixture(scope="module", autouse=True)
def enable_poplibs_logging():
    with patch.dict("os.environ", {"POPLIBS_LOG_LEVEL": "DEBUG"}):
        yield


def available_memory_proportion_harness(capfd, insert_operator,
                                        avail_mem_prop):
    builder = popart.Builder()

    input_shape = popart.TensorInfo("FLOAT", [1, 2, 4, 4])
    weight_shape = popart.TensorInfo("FLOAT", [3, 2, 4, 4])

    weight_data = np.ones(weight_shape.shape(), np.float32)
    input_ = builder.addInputTensor(input_shape)
    weights = builder.addInitializedInputTensor(weight_data)
    act = insert_operator(builder, input_, weights, avail_mem_prop)
    o = builder.aiOnnx.relu([act])
    loss = builder.aiGraphcore.identityloss([o])

    anchor_names = [
        o,
        popart.reservedGradientPrefix() + input_,
        popart.reservedGradientPrefix() + weights
    ]
    training_dataFlow = popart.DataFlow(
        1, {
            anchor_names[0]: popart.AnchorReturnType("All"),
            anchor_names[1]: popart.AnchorReturnType("All"),
            anchor_names[2]: popart.AnchorReturnType("All")
        })

    opts = popart.SessionOptions()
    opts.constantWeights = False  # Allow the weights to be updated

    # Create the device
    device = tu.create_test_device(1, opts={"compileIPUCode": True})
    device.attach()

    # Prepare the input data
    input_data = np.random.random_sample(input_shape.shape()).astype(
        np.float32)

    # Prepare the Training session
    training_session = popart.TrainingSession(fnModel=builder.getModelProto(),
                                              dataFlow=training_dataFlow,
                                              loss=loss,
                                              optimizer=popart.ConstSGD(0.01),
                                              userOptions=opts,
                                              deviceInfo=device)

    # Compile the training graph
    training_session.prepareDevice()

    # Run the training session
    training_session.weightsFromHost()

    training_anchors = training_session.initAnchorArrays()
    training_inputs = {input_: input_data}

    training_session.run(popart.PyStepIO(training_inputs, training_anchors))

    captured = capfd.readouterr()

    return captured.err


def assert_contains(pattern, output):
    # Find the regex matches.
    matches = re.findall(pattern, output)
    assert len(
        matches
    ) > 0, f"Failed to find pattern {pattern} in log output:\n\n{output}"


def insert_conv(builder, inputs, weights, avail_mem_prop):
    act = builder.aiOnnx.conv([inputs, weights],
                              dilations=[1, 1],
                              pads=[1, 1, 1, 1],
                              strides=[1, 1])
    builder.setAvailableMemoryProportion(act, avail_mem_prop)
    return act


def insert_matmul(builder, inputs, weights, avail_mem_prop):
    act = builder.aiOnnx.matmul([inputs, weights])
    builder.setAvailableMemoryProportion(act, avail_mem_prop)
    return act


def insert_gather(builder, inputs, weights, avail_mem_prop):
    ind = np.array([1, 0]).astype(np.float32)
    indices = builder.addInitializedInputTensor(ind)
    x = builder.aiOnnx.gather([inputs, indices], axis=1)
    builder.setAvailableMemoryProportion(x, avail_mem_prop)
    act = builder.aiOnnx.matmul([x, weights])
    return act


def insert_scatter_reduce(builder, inputs, weights, avail_mem_prop):
    np.random.seed(0)
    ind = np.random.randint(0, high=4, size=[1, 2, 4, 4]).astype(np.float32)
    indices = builder.addInitializedInputTensor(ind)
    x = builder.aiGraphcore.scatterreduce([inputs, indices], axis_size=4)
    builder.setAvailableMemoryProportion(x, avail_mem_prop)
    act = builder.aiOnnx.matmul([x, weights])
    return act


def insert_lstm(builder, inputs, weights, avail_mem_prop):
    np.random.seed(0)
    d1 = np.array([[[1., 2., 3.], [4., 5., 6.]],
                   [[7., 8., 9.], [10., 11., 12.]]]).astype(np.float32)

    input_size = d1.shape[2]
    hidden_size = 7

    d2 = np.random.rand(1, 4 * hidden_size, input_size).astype(np.float32)
    d3 = np.zeros((1, 4 * hidden_size, hidden_size)).astype(np.float32)

    i1 = builder.addInitializedInputTensor(d1)
    i2 = builder.addInitializedInputTensor(d2)
    i3 = builder.addInitializedInputTensor(d3)

    Y, Y_h, Y_c = builder.aiOnnx.lstm([i1, i2, i3], 3, clip=None)
    builder.setAvailableMemoryProportion({Y, Y_h, Y_c}, avail_mem_prop)

    act = builder.aiOnnx.matmul([inputs, weights])
    shape = builder.aiOnnx.constant(np.array([3, 2, 2, 8]))
    act = builder.aiOnnx.reshape([act, shape])
    starts = builder.aiOnnx.constant(np.array([0, 0, 0, 0]))
    ends = builder.aiOnnx.constant(np.array([2, 1, 2, 7]))
    act = builder.aiOnnx.slice([act, starts, ends])
    act = builder.aiOnnx.add([act, Y])
    return act


# Test that poplar gets our instruction to set the available memory proportion.
# Do this by matching the poplibs logs.
@tu.requires_ipu_model
def test_conv_avail_memory_log(capfd):
    avail_mem_prop = 0.6
    output = available_memory_proportion_harness(capfd, insert_conv,
                                                 avail_mem_prop)

    # This is the available tile memory for the conv.
    # TODO: Update this if future chips have more memory per tile.
    avail_mem = int(np.floor(avail_mem_prop * 638976))
    patt = f"Planning convolution with a per-tile memory limit of {avail_mem}"
    assert_contains(patt, output)


# Test outside [0,1) error which is thrown by Builder::setAvailableMemoryProportion
@tu.requires_ipu_model
def test_conv_avail_memory_error(capfd):

    avail_mem_prop = 1.1  # Wrong value

    with pytest.raises(popart.popart_exception) as e_info:
        available_memory_proportion_harness(capfd, insert_conv, avail_mem_prop)

    assert (e_info.value.args[0].startswith(
        "availableMemoryProportion must be in (0,1]"))


# Test that poplar gets our instruction to set the available memory proportion.
# Do this by matching the poplibs logs.
@tu.requires_ipu_model
def test_matmul_avail_memory_log(capfd):
    avail_mem_prop = 0.6
    output = available_memory_proportion_harness(capfd, insert_matmul,
                                                 avail_mem_prop)

    # This is the available tile memory for the matmul.
    # TODO: Update this if future chips have more memory per tile.
    avail_mem = int(np.floor(avail_mem_prop * 638976))
    patt = f"Planning convolution with a per-tile memory limit of {avail_mem}"
    assert_contains(patt, output)


@tu.requires_ipu_model
def test_gather_avail_memory_log(capfd):
    avail_mem_prop = 0.5
    output = available_memory_proportion_harness(capfd, insert_gather,
                                                 avail_mem_prop)
    pattern = f"availableMemoryProportion={avail_mem_prop:0.1f}"
    assert_contains(pattern, output)


@tu.requires_ipu_model
def test_scatter_reduce_avail_memory_log(capfd):
    avail_mem_prop = 0.6
    output = available_memory_proportion_harness(capfd, insert_scatter_reduce,
                                                 avail_mem_prop)
    pattern = f"availableMemoryProportion={avail_mem_prop:0.1f}"
    assert_contains(pattern, output)


@tu.requires_ipu_model
def test_lstm_avail_memory_log(capfd):
    avail_mem_prop = 0.4
    output = available_memory_proportion_harness(capfd, insert_lstm,
                                                 avail_mem_prop)
    # This is the available tile memory for the matmul.
    # TODO: Update this if future chips have more memory per tile.
    avail_mem = int(np.floor(avail_mem_prop * 638976))
    patt = f"Planning convolution with a per-tile memory limit of {avail_mem}"
    assert_contains(patt, output)
