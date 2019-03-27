import poponnx
import pytest
import numpy as np
import test_util as tu


# Test that you can train a model and then use the weight in a inference run
def test_train_then_infer_via_file():

    builder = poponnx.Builder()

    input_shape = poponnx.TensorInfo("FLOAT", [1, 2, 4, 4])
    weight_shape = poponnx.TensorInfo("FLOAT", [3, 2, 3, 3])

    weight_data = np.ones([3, 2, 3, 3], np.float32)
    input = builder.addInputTensor(input_shape)
    weights = builder.addInitializedInputTensor(weight_data)
    act = builder.aiOnnx.conv([input, weights],
                              dilations=[1, 1],
                              pads=[1, 1, 1, 1],
                              strides=[1, 1])
    o = builder.aiOnnx.relu([act])

    builder.addOutputTensor(o)

    anchor_names = [o, 'd__' + input, 'd__' + weights]
    training_dataFlow = poponnx.DataFlow(
        1, {
            anchor_names[0]: poponnx.AnchorReturnType("ALL"),
            anchor_names[1]: poponnx.AnchorReturnType("ALL"),
            anchor_names[2]: poponnx.AnchorReturnType("ALL")
        })

    opts = poponnx.SessionOptions()
    opts.constantWeights = False  # Allow the weights to be updated

    # ----------------------------------------------

    # Create the device
    options = {"compileIPUCode": True, 'numIPUs': 1, "tilesPerIPU": 1216}
    device = poponnx.DeviceManager().createIpuModelDevice(options)
    device.attach()

    # ----------------------------------------------

    # Prepare the input data
    input_data = np.ones(input_shape.shape(), dtype=np.float32)

    # ----------------------------------------------

    # Prepare the Inference session
    inference_dataFlow = poponnx.DataFlow(1,
                                          {o: poponnx.AnchorReturnType("ALL")})

    inference_session = poponnx.InferenceSession(
        fnModel=builder.getModelProto(),
        dataFeed=inference_dataFlow,
        userOptions=opts)

    # Compile the inference graph
    inference_session.setDevice(device)
    inference_session.prepareDevice()

    # ----------------------------------------------

    # Prepare the Training session
    training_session = poponnx.TrainingSession(
        fnModel=builder.getModelProto(),
        dataFeed=training_dataFlow,
        losses=[poponnx.L1Loss(o, "l1LossVal", 0.1)],
        optimizer=poponnx.ConstSGD(0.01),
        userOptions=opts)

    # Compile the training graph
    training_session.setDevice(device)
    training_session.prepareDevice()

    # ----------------------------------------------

    # Run the training session
    training_session.weightsFromHost()
    training_session.optimizerFromHost()

    training_anchors = training_session.initAnchorArrays()
    training_inputs = {input: input_data}

    for i in range(4):
        training_session.run(
            poponnx.PyStepIO(training_inputs, training_anchors))

    # Save the trained weights
    training_session.modelToHost("test.onnx")

    # ----------------------------------------------

    # Run the inference session
    ## Load the updated weights from the training session
    inference_session.resetHostWeights("test.onnx")
    inference_session.weightsFromHost()

    inference_anchors = inference_session.initAnchorArrays()
    inference_inputs = {input: input_data}

    inference_session.run(
        poponnx.PyStepIO(inference_inputs, inference_anchors))
