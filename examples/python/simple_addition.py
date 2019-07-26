import numpy as np
import popart

# Create a builder and construct a graph
builder = popart.Builder()

data_shape = popart.TensorInfo("FLOAT", [1])

a = builder.addInputTensor(data_shape)
b = builder.addInputTensor(data_shape)

o = builder.aiOnnx.add([a, b])

builder.addOutputTensor(o)

proto = builder.getModelProto()

# Describe how to run the model
dataFlow = popart.DataFlow(1, {o: popart.AnchorReturnType("ALL")})

# Create a session to compile and execute the graph
session = popart.InferenceSession(
    fnModel=proto,
    dataFeed=dataFlow,
    deviceInfo=popart.DeviceManager().createIpuModelDevice({}))

# Compile graph
session.prepareDevice()

# Create buffers to receive results from the execution
anchors = session.initAnchorArrays()

# Generate some random input data
data_a = np.random.rand(1).astype(np.float32)
data_b = np.random.rand(1).astype(np.float32)

stepio = popart.PyStepIO({a: data_a, b: data_b}, anchors)
session.run(stepio)

print("Input a is " + str(data_a))
print("Input b is " + str(data_b))
print("Result is " + str(anchors[o]))
