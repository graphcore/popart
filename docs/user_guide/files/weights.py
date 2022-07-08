# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
"""
Demonstrate how to read the weight from a training session and use them in an inference session.
"""

import numpy as np
import popart

# Create a builder and construct a graph
# ------------------------------------------------------------------------------

builder = popart.Builder()

data_shape = popart.TensorInfo("FLOAT16", [1, 2])
lbl_shape = popart.TensorInfo("INT32", [1])

ip = builder.addInputTensor(data_shape)
lb = builder.addInputTensor(lbl_shape)

w = builder.addInitializedInputTensor(np.ones([2, 2], np.float16))
b = builder.addInitializedInputTensor(np.ones([2], np.float16))
o = builder.aiOnnx.gemm([ip, w, b], 1.0, 1.0, False, False)
o = builder.aiOnnx.relu([o])
o = builder.aiOnnx.softmax([o])
o = builder.aiGraphcore.nllloss([o, lb])

dataFlow = popart.DataFlow(1, {o: popart.AnchorReturnType("All")})

# Create a session to compile and the graph for inference
# ------------------------------------------------------------------------------
inferenceOptions = popart.SessionOptions()
# Need to compile the inference graph with variable weights we they can be updated
# before execution
inferenceOptions.constantWeights = False

inferenceSession = popart.InferenceSession(
    fnModel=builder.getModelProto(),
    dataFlow=dataFlow,
    userOptions=inferenceOptions,
    deviceInfo=popart.DeviceManager().createIpuModelDevice({}),
)

# Compile graph
inferenceSession.prepareDevice()

# Create buffers to receive results from the execution
inferenceAnchors = inferenceSession.initAnchorArrays()

# Create a session to compile and the graph for training
# ------------------------------------------------------------------------------
trainingOptions = popart.SessionOptions()
trainingSession = popart.TrainingSession(
    fnModel=builder.getModelProto(),
    dataFlow=dataFlow,
    loss=o,
    optimizer=popart.ConstSGD(0.001),
    userOptions=trainingOptions,
    deviceInfo=popart.DeviceManager().createIpuModelDevice({}),
)

# Compile graph
trainingSession.prepareDevice()

# Execute the training graph
# ------------------------------------------------------------------------------

# Generate some random input data
trainingData = np.random.rand(1, 2).astype(np.float16)
trainingDataLabels = np.random.rand(1).astype(np.int32)

# Create buffers to receive results from the execution
trainingAnchors = trainingSession.initAnchorArrays()
trainingStepio = popart.PyStepIO(
    {ip: trainingData, lb: trainingDataLabels}, trainingAnchors
)

# Copy the weights to the device from the host
trainingSession.weightsFromHost()

# Run the training graph
trainingSession.run(trainingStepio)

# Copy the weights to the host from the device
trainingSession.weightsToHost()

# Prepare the map of weights to read the weights into
weights = {}
weights[w] = np.empty([2, 2], np.float16)
weightsIo = popart.PyWeightsIO(weights)

# Read the weights from the session
trainingSession.readWeights(weightsIo)

# Execute the inference graph
# ------------------------------------------------------------------------------

# Generate some random input data
interenceData = np.random.rand(1, 2).astype(np.float16)
interenceDataLabels = np.random.rand(1).astype(np.int32)

# Create buffers to receive results from the execution
inferenceAnchors = inferenceSession.initAnchorArrays()
inferenceStepio = popart.PyStepIO(
    {ip: interenceData, lb: interenceDataLabels}, inferenceAnchors
)

# Write weights to the session
inferenceSession.writeWeights(weightsIo)

# Copy the weights to the device from the host
inferenceSession.weightsFromHost()

# Run the inference graph
inferenceSession.run(inferenceStepio)
