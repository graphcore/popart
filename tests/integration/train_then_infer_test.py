# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import popart
import pytest
import numpy as np
import test_util as tu
import tempfile
import os


# Test that you can train a model and then use the weight in a inference run
@tu.requires_ipu_model
def test_train_then_infer_via_file():

    builder = popart.Builder()

    input_shape = popart.TensorInfo("FLOAT", [1, 2, 4, 4])
    weight_shape = popart.TensorInfo("FLOAT", [3, 2, 3, 3])

    weight_data = np.ones([3, 2, 3, 3], np.float32)
    input = builder.addInputTensor(input_shape)
    weights = builder.addInitializedInputTensor(weight_data)
    act = builder.aiOnnx.conv([input, weights],
                              dilations=[1, 1],
                              pads=[1, 1, 1, 1],
                              strides=[1, 1])
    o = builder.aiOnnx.relu([act])

    l1 = builder.aiGraphcore.l1loss([o], 0.1)

    anchor_names = [
        o,
        popart.reservedGradientPrefix() + input,
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
    tempDir = tempfile.TemporaryDirectory()
    opts.engineOptions["autoReport.directory"] = tempDir.name
    opts.engineOptions["autoReport.all"] = "true"

    # ----------------------------------------------

    # Create the device
    device = tu.create_test_device(1, opts={"compileIPUCode": True})
    device.attach()

    # ----------------------------------------------

    # Prepare the input data
    input_data = np.ones(input_shape.shape(), dtype=np.float32)

    # ----------------------------------------------

    # Prepare the Inference session
    inference_dataFlow = popart.DataFlow(1,
                                         {o: popart.AnchorReturnType("All")})

    inference_session = popart.InferenceSession(
        fnModel=builder.getModelProto(),
        dataFlow=inference_dataFlow,
        userOptions=opts,
        deviceInfo=device)

    # Compile the inference graph
    inference_session.prepareDevice()

    # ----------------------------------------------

    # Prepare the Training session
    training_session = popart.TrainingSession(fnModel=builder.getModelProto(),
                                              dataFlow=training_dataFlow,
                                              loss=l1,
                                              optimizer=popart.ConstSGD(0.01),
                                              userOptions=opts,
                                              deviceInfo=device,
                                              name="ivor")

    # Compile the training graph
    training_session.prepareDevice()

    # ----------------------------------------------

    # Run the training session
    training_session.weightsFromHost()

    training_anchors = training_session.initAnchorArrays()
    training_inputs = {input: input_data}

    for i in range(4):
        training_session.run(popart.PyStepIO(training_inputs,
                                             training_anchors))

    # Save the trained weights
    training_session.modelToHost("test.onnx")

    # ----------------------------------------------

    # Run the inference session
    ## Load the updated weights from the training session
    inference_session.resetHostWeights("test.onnx")
    inference_session.weightsFromHost()

    inference_anchors = inference_session.initAnchorArrays()
    inference_inputs = {input: input_data}

    inference_session.run(popart.PyStepIO(inference_inputs, inference_anchors))

    # check that the profile.pop as been created in the subdirectories
    assert (os.path.isfile(tempDir.name + "/inference/profile.pop"), True)
    assert (os.path.isfile(tempDir.name + "/ivor/profile.pop"), True)


@tu.requires_ipu_model
def test_cannot_call_resethostweights_with_constant_weights():

    builder = popart.Builder()

    input_shape = popart.TensorInfo("FLOAT", [1, 2, 4, 4])

    weight_data = np.ones([3, 2, 3, 3], np.float32)
    input = builder.addInputTensor(input_shape)
    weights = builder.addInitializedInputTensor(weight_data)
    act = builder.aiOnnx.conv([input, weights],
                              dilations=[1, 1],
                              pads=[1, 1, 1, 1],
                              strides=[1, 1])
    o = builder.aiOnnx.relu([act])

    builder.addOutputTensor(o)

    opts = popart.SessionOptions()
    opts.constantWeights = True  # Fix weights in inference session

    # ----------------------------------------------

    # Create the device
    device = tu.create_test_device(1, opts={"compileIPUCode": True})
    device.attach()

    # ----------------------------------------------

    # Prepare the input data
    input_data = np.ones(input_shape.shape(), dtype=np.float32)

    # ----------------------------------------------

    # Prepare the Inference session
    inference_dataFlow = popart.DataFlow(1,
                                         {o: popart.AnchorReturnType("All")})

    inference_session = popart.InferenceSession(
        fnModel=builder.getModelProto(),
        dataFlow=inference_dataFlow,
        userOptions=opts,
        deviceInfo=device)

    # Compile the inference graph
    inference_session.prepareDevice()

    # Create a file with some weights
    inference_session.modelToHost("test.onnx")

    ## Load the updated weights from the training session
    with pytest.raises(popart.popart_exception) as e_info:
        inference_session.resetHostWeights("test.onnx")

    assert (e_info.value.args[0].startswith(
        "Cannot call resetHostWeights when constantWeights is set"))


@tu.requires_ipu_model
def test_modelToHost_calls_resetHostWeights():
    builder = popart.Builder()

    input_shape = popart.TensorInfo("FLOAT", [1, 1, 4, 4])
    weight_shape = popart.TensorInfo("FLOAT", [1, 1, 3, 3])

    input_data = np.ones(input_shape.shape(), dtype=np.float32)
    weight_data = np.ones(weight_shape.shape(), np.float32)

    input = builder.addInputTensor(input_shape)
    weights = builder.addInitializedInputTensor(weight_data)

    act = builder.aiOnnx.conv([input, weights],
                              dilations=[1, 1],
                              pads=[1, 1, 1, 1],
                              strides=[1, 1])
    o = builder.aiOnnx.relu([act])
    l1 = builder.aiGraphcore.l1loss([o], 0.1)

    builder.addOutputTensor(o)

    anchor_names = [o]
    data_flow = popart.DataFlow(
        1, {i: popart.AnchorReturnType("All")
            for i in anchor_names})

    opts = popart.SessionOptions()
    opts.constantWeights = False  # Allow the weights to be updated

    # Create the device
    device = tu.create_test_device(1, opts={"compileIPUCode": True})
    device.attach()

    # Prepare the Training session
    session = popart.TrainingSession(fnModel=builder.getModelProto(),
                                     dataFlow=data_flow,
                                     loss=l1,
                                     optimizer=popart.ConstSGD(0.1),
                                     userOptions=opts,
                                     deviceInfo=device)

    # Compile the training graph
    session.prepareDevice()

    session.weightsFromHost()

    anchors = session.initAnchorArrays()
    inputs = {input: input_data}

    outputs = []

    for i in range(2):
        session.run(popart.PyStepIO(inputs, anchors))
        outputs.append(np.copy(anchors[o]))

    # The outputs of the two training runs should not be close
    print('Checking first two outputs differ')
    assert not np.allclose(outputs[0], outputs[1])

    # Write weights from device to host
    session.modelToHost("test.onnx")

    # Write weights from host to device
    session.weightsFromHost()

    # Run the training session and get the output
    session.run(popart.PyStepIO(inputs, anchors))
    outputs.append(np.copy(anchors[o]))

    # Neither of the previous outputs should be close to the new output
    print('Checking third output differs from first two')
    assert not np.allclose(outputs[2], outputs[0])
    assert not np.allclose(outputs[2], outputs[1])

    # Last output should be close to second output minus
    # the difference between first and second outputs.
    delta_outputs = outputs[0] - outputs[1]
    expected_out = outputs[1] - delta_outputs
    print('Checking third output is close to expected value')
    assert np.allclose(outputs[2], expected_out)
