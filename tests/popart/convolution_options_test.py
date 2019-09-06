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

    # Apply the setAvailableMemoryProportion to the convolution
    if apply_to_conv:
        builder.setAvailableMemoryProportion(act, avail_mem_prop)
    # For the test_conv_avail_memory_error_2 test we try to apply the
    # setAvailableMemoryProportion to the relu op defined above, rather
    # than the expected convolution op, and expect an error.
    else:
        builder.setAvailableMemoryProportion(o, avail_mem_prop)

    builder.addOutputTensor(o)

    anchor_names = [
        o,
        popart.reservedGradientPrefix() + input_,
        popart.reservedGradientPrefix() + weights
    ]
    training_dataFlow = popart.DataFlow(
        1, {
            anchor_names[0]: popart.AnchorReturnType("ALL"),
            anchor_names[1]: popart.AnchorReturnType("ALL"),
            anchor_names[2]: popart.AnchorReturnType("ALL")
        })

    opts = popart.SessionOptions()
    opts.constantWeights = False  # Allow the weights to be updated

    # Create the device
    options = {"compileIPUCode": True, 'numIPUs': 1, "tilesPerIPU": 1216}
    device = popart.DeviceManager().createIpuModelDevice(options)
    device.attach()

    # Prepare the input data
    input_data = np.random.random_sample(input_shape.shape()).astype(
        np.float32)

    # Prepare the Training session
    training_session = popart.TrainingSession(
        fnModel=builder.getModelProto(),
        dataFeed=training_dataFlow,
        losses=[popart.L1Loss(o, "l1LossVal", 0.1)],
        optimizer=popart.ConstSGD(0.01),
        userOptions=opts,
        deviceInfo=device)

    # Compile the training graph
    training_session.prepareDevice()

    # Run the training session
    training_session.weightsFromHost()
    training_session.optimizerFromHost()

    training_anchors = training_session.initAnchorArrays()
    training_inputs = {input_: input_data}

    training_session.run(popart.PyStepIO(training_inputs, training_anchors))

    captured = capfd.readouterr()
    os.environ["POPLIBS_LOG_LEVEL"] = "NONE"

    return captured.err


# Test that poplar gets our instruction to set the available memory proportion.
# Do this by matching the poplibs logs.
def test_conv_avail_memory_log(tmpdir, capfd):

    avail_mem_prop = 0.6
    output = conv_avail_memory(tmpdir, capfd, True, avail_mem_prop)

    # This is the available tile memory for the conv.
    # TODO: Update this if future chips have more memory per tile.
    avail_mem = int(np.floor(avail_mem_prop * 262_144))
    patt = f"Planning convolution with a per-tile memory limit of {avail_mem}"

    # Find the regex matches.
    matches = re.findall(patt, output)
    assert len(matches) > 1


# Test outside [0,1) error
def test_conv_avail_memory_error(tmpdir, capfd):

    avail_mem_prop = 1.1  # Wrong value

    with pytest.raises(popart.popart_exception) as e_info:
        conv_avail_memory(tmpdir, capfd, True, avail_mem_prop)

    assert (e_info.value.args[0].startswith(
        "availableMemoryProportion must be in (0,1]"))


# Test wrong op error
def test_conv_avail_memory_error_2(tmpdir, capfd):

    avail_mem_prop = 0.6

    # Apply to the wrong op
    with pytest.raises(popart.popart_exception) as e_info:
        conv_avail_memory(tmpdir, capfd, False, avail_mem_prop)

    assert (e_info.value.args[0].startswith(
        "Builder::setAvailableMemoryProportion should only be called on Conv"))
