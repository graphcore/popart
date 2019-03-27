import numpy as np
import poponnx

# Create a builder and construct a graph
builder = poponnx.Builder()

data_shape = poponnx.TensorInfo("FLOAT", [1])

a = builder.addInputTensor(data_shape)
b = builder.addInputTensor(data_shape)

o = builder.aiOnnx.add([a, b])

builder.addOutputTensor(o)

proto = builder.getModelProto()

# Describe how to run the model
dataFlow = poponnx.DataFlow(1, {o: poponnx.AnchorReturnType("ALL")})

# Create a session to compile and execute the graph
session = poponnx.InferenceSession(
    fnModel=proto,
    dataFeed=dataFlow,
    deviceInfo=poponnx.DeviceManager().createIpuModelDevice({}))

# Compile graph
session.prepareDevice()

# Create buffers to receive results from the execution
anchors = session.initAnchorArrays()

# Generate some random input data
data_a = np.random.rand(1).astype(np.float32)
data_b = np.random.rand(1).astype(np.float32)

stepio = poponnx.PyStepIO({a: data_a, b: data_b}, anchors)
session.run(stepio)

print("Input a is " + str(data_a))
print("Input b is " + str(data_b))
print("Result is " + str(anchors[o]))
