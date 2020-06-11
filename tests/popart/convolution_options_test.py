# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import json
import os
import re

import numpy as np
import pytest

import popart
import test_util as tu


# Standard conv and relu with setAvailableMemoryProportion setting
def conv_avail_memory(tmpdir, capfd, apply_to_conv=True, avail_mem_prop=0.9):
    os.environ["POPLIBS_LOG_LEVEL"] = "DEBUG"

    builder = popart.Builder()

    input_shape = popart.TensorInfo("FLOAT", [1, 2, 4, 4])
    weight_shape = popart.TensorInfo("FLOAT", [3, 2, 3, 3])

    weight_data = np.ones(weight_shape.shape(), np.float32)
    input_ = builder.addInputTensor(input_shape)
    weights = builder.addInitializedInputTensor(weight_data)
    act = builder.aiOnnx.conv([input_, weights],
                              dilations=[1, 1],
                              pads=[1, 1, 1, 1],
                              strides=[1, 1])
    o = builder.aiOnnx.relu([act])
    loss = builder.aiGraphcore.identityloss([o])

    # Apply the setAvailableMemoryProportion to the convolution
    if apply_to_conv:
        builder.setAvailableMemoryProportion(act, avail_mem_prop)
    # For the test_conv_avail_memory_error_2 test we try to apply the
    # setAvailableMemoryProportion to the relu op defined above, rather
    # than the expected convolution op, and expect an error.
    else:
        builder.setAvailableMemoryProportion(o, avail_mem_prop)

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
    os.environ["POPLIBS_LOG_LEVEL"] = "NONE"

    return captured.err


# Standard matmul and relu with setAvailableMemoryProportion setting
def matmul_avail_memory(tmpdir, capfd, apply_to_conv=True, avail_mem_prop=0.9):
    os.environ["POPLIBS_LOG_LEVEL"] = "DEBUG"

    builder = popart.Builder()

    input_shape = popart.TensorInfo("FLOAT", [2, 4])
    weight_shape = popart.TensorInfo("FLOAT", [4, 8])

    weight_data = np.ones(weight_shape.shape(), np.float32)
    input_ = builder.addInputTensor(input_shape)
    weights = builder.addInitializedInputTensor(weight_data)
    act = builder.aiOnnx.matmul([input_, weights])
    o = builder.aiOnnx.relu([act])
    loss = builder.aiGraphcore.identityloss([o])

    # Apply the setAvailableMemoryProportion to the matmul
    if apply_to_conv:
        builder.setAvailableMemoryProportion(act, avail_mem_prop)
    # For the test_conv_avail_memory_error_2 test we try to apply the
    # setAvailableMemoryProportion to the relu op defined above, rather
    # than the expected convolution op, and expect an error.
    else:
        builder.setAvailableMemoryProportion(o, avail_mem_prop)

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
    os.environ["POPLIBS_LOG_LEVEL"] = "NONE"

    return captured.err


# Test that poplar gets our instruction to set the available memory proportion.
# Do this by matching the poplibs logs.
@tu.requires_ipu_model
def test_conv_avail_memory_log(tmpdir, capfd):

    avail_mem_prop = 0.6
    output = conv_avail_memory(tmpdir, capfd, True, avail_mem_prop)

    # This is the available tile memory for the conv.
    # TODO: Update this if future chips have more memory per tile.
    avail_mem = int(np.floor(avail_mem_prop * 262_144))
    patt = f"Planning convolution with a per-tile memory limit of {avail_mem}"

    # Find the regex matches.
    matches = re.findall(patt, output)
    print(output)
    print(matches)
    assert len(matches) > 1


# Test outside [0,1) error
@tu.requires_ipu_model
def test_conv_avail_memory_error(tmpdir, capfd):

    avail_mem_prop = 1.1  # Wrong value

    with pytest.raises(popart.popart_exception) as e_info:
        conv_avail_memory(tmpdir, capfd, True, avail_mem_prop)

    assert (e_info.value.args[0].startswith(
        "availableMemoryProportion must be in (0,1]"))


# Test wrong op error
@tu.requires_ipu_model
def test_avail_memory_error_2(tmpdir, capfd):

    avail_mem_prop = 0.6

    # Apply to the wrong op
    with pytest.raises(popart.popart_exception) as e_info:
        conv_avail_memory(tmpdir, capfd, False, avail_mem_prop)

    assert (e_info.value.args[0].startswith(
        "Builder::setAvailableMemoryProportion should only be called on Conv or MatMul"
    ))


# Test that poplar gets our instruction to set the available memory proportion.
# Do this by matching the poplibs logs.
@tu.requires_ipu_model
def test_matmul_avail_memory_log(tmpdir, capfd):

    avail_mem_prop = 0.6
    output = matmul_avail_memory(tmpdir, capfd, True, avail_mem_prop)

    # This is the available tile memory for the matmul.
    # TODO: Update this if future chips have more memory per tile.
    avail_mem = int(np.floor(avail_mem_prop * 262_144))
    patt = f"Planning convolution with a per-tile memory limit of {avail_mem}"

    # Find the regex matches.
    matches = re.findall(patt, output)
    print(output)
    print(matches)
    assert len(matches) > 1
