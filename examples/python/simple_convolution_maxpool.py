# An example of a forward pass where an input is passed through a convolution
# and a maxpool.

import numpy as np
import poponnx

# Create a builder to construct the graph
builder = poponnx.Builder()

# Define the shape of the input
data_shape = poponnx.TensorInfo("FLOAT", [1, 4, 24, 24])

# Add an input tensor for the data input
ip = builder.addInputTensor(data_shape)

# Add a weights and a bias tensor
weights = builder.addInitializedInputTensor(np.zeros([2, 4, 3, 3], np.float32))
biases = builder.addInitializedInputTensor(np.zeros([2], np.float32))

# Add the convolution
o = builder.convolution([ip, weights, biases], [1, 1], [1, 1, 1, 1], [1, 1], 1,
                        True, "Layer_1")

# Add a 3x3 maxpooling layer
o = builder.maxpool([o], [3, 3], [2, 2], [1, 1, 1, 1])

# Note which is the output tensor
builder.addOutputTensor(o)

# Generate an ONNX model protobuf
proto = builder.getModelProto()

# Describe how to run the model
dataFlow = poponnx.DataFlow(1, 1, {o: poponnx.AnchorReturnType("ALL")})

# Reporting options
opts = poponnx.SessionOptions()
opts.reportOptions = {'doLayerWiseBreakdown': 'true'}

# Create a session to compile and execute the graph
session = poponnx.Session(
    fnModel=proto, dataFeed=dataFlow, outputdir=".", userOptions=opts)

# Set the device to an IPU device
session.setDevice(poponnx.DeviceManager().createIpuModelDevice({}))

# Compile the graph and prepare for execution
session.prepareDevice()

# Create buffers to receive results from the execution
anchors = session.initAnchorArrays()

# Generate some random input data
data = np.random.rand(1, 4, 24, 24).astype(np.float32)

stepio = poponnx.PyStepIO({ip: data}, anchors)
session.infer(stepio)

print("Result is " + str(anchors[o]))

print("Report")
print(session.getSummaryReport())
