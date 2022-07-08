# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import numpy as np
import pytest
import re
from unittest.mock import patch

import popart
import test_util as tu


@pytest.fixture(scope="module", autouse=True)
def enable_poplibs_logging():
    with patch.dict("os.environ", {"POPLIBS_LOG_LEVEL": "DEBUG"}):
        yield


def conv_settings(capfd, operation):
    builder = popart.Builder()

    input_shape = popart.TensorInfo("FLOAT", [1, 2, 4, 4])
    weight_shape = popart.TensorInfo("FLOAT", [3, 2, 3, 3])

    weight_data = np.ones(weight_shape.shape(), np.float32)
    input_ = builder.addInputTensor(input_shape)
    weights = builder.addInitializedInputTensor(weight_data)
    act = builder.aiOnnx.conv(
        [input_, weights], dilations=[1, 1], pads=[1, 1, 1, 1], strides=[1, 1]
    )
    o = builder.aiOnnx.relu([act])
    loss = builder.aiGraphcore.identityloss([o])

    operation(builder, act=act, o=o)

    anchor_names = [
        o,
        popart.reservedGradientPrefix() + input_,
        popart.reservedGradientPrefix() + weights,
    ]
    training_dataFlow = popart.DataFlow(
        1,
        {
            anchor_names[0]: popart.AnchorReturnType("All"),
            anchor_names[1]: popart.AnchorReturnType("All"),
            anchor_names[2]: popart.AnchorReturnType("All"),
        },
    )

    opts = popart.SessionOptions()
    opts.constantWeights = False  # Allow the weights to be updated

    # Create the device
    with tu.create_test_device(1, opts={"compileIPUCode": True}) as device:
        device.attach()

        # Prepare the input data
        input_data = np.random.random_sample(input_shape.shape()).astype(np.float32)

        # Prepare the Training session
        training_session = popart.TrainingSession(
            fnModel=builder.getModelProto(),
            dataFlow=training_dataFlow,
            loss=loss,
            optimizer=popart.ConstSGD(0.01),
            userOptions=opts,
            deviceInfo=device,
        )

        # Compile the training graph
        training_session.prepareDevice()

        # Run the training session
        training_session.weightsFromHost()

        training_anchors = training_session.initAnchorArrays()
        training_inputs = {input_: input_data}

        training_session.run(popart.PyStepIO(training_inputs, training_anchors))

    captured = capfd.readouterr()

    return captured.err


# Standard conv and relu with setEnableConvDithering setting
def conv_enable_dithering(capfd, apply_to_conv=True, dithering=True):
    def operation(builder, **kwargs):
        # Apply the setEnableConvDithering to the convolution
        if apply_to_conv:
            builder.setEnableConvDithering(kwargs.get("act", None), dithering)
        # For the test_conv_dithering_error_2 test we try to apply the
        # setEnableConvDithering to the relu op defined above, rather
        # than the expected convolution op, and expect an error.
        else:
            builder.setEnableConvDithering(kwargs.get("o", None), dithering)

    return conv_settings(capfd, operation=operation)


@tu.requires_ipu_model
@pytest.mark.parametrize(
    "apply_to_conv, dithering, exception_string",
    [
        (
            False,
            True,
            "Builder::setEnableConvDithering should only be called on convolutions but was given: Relu",
        ),
        (True, False, None),
        (True, True, None),
        (True, 44, "enableConvDithering must be a bool value"),
    ],
)
def test_enable_conv_dithering_error(capfd, apply_to_conv, dithering, exception_string):
    if exception_string is not None and isinstance(exception_string, str):
        with pytest.raises(popart.popart_exception) as e_info:
            conv_enable_dithering(
                capfd, apply_to_conv=apply_to_conv, dithering=dithering
            )
        assert e_info.value.args[0] == exception_string
    else:
        output = conv_enable_dithering(
            capfd, apply_to_conv=apply_to_conv, dithering=dithering
        )
        matches = re.findall(" +enableConvDithering +" + str(int(dithering)), output)
        assert len(matches) > 1
        matches = re.findall(
            " +enableConvDithering +" + str(int(not dithering)), output
        )
        assert len(matches) == 0
