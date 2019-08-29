import numpy as np
import popart
# import onnx
import ctypes
import os

# Load the .so file in the same directory.
so_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                       "custom_op.so")
custom_ops = ctypes.cdll.LoadLibrary(so_path)

# Create a builder and construct a graph
builder = popart.Builder()

data_shape = popart.TensorInfo("FLOAT16", [1, 2])
lbl_shape = popart.TensorInfo("INT32", [1])

a = builder.addInputTensor(data_shape)
lb = builder.addInputTensor(lbl_shape)
# Create some weights
w = builder.addInitializedInputTensor(np.ones([2, 2], np.float16))
b = builder.addInitializedInputTensor(np.ones([2], np.float16))

# We have an unknown number of output tensors.
# So we have to reference the zero'th tensor in this example.
o = builder.customOp(opName="Cube", opVersion=1, domain="ai.acme",
                     inputs=[a])[0]

o = builder.aiOnnx.gemm([a, w, b], 1., 1., False, False)
o = builder.aiOnnx.relu([o])
o = builder.aiOnnx.softmax([o])
builder.addOutputTensor(o)

proto = builder.getModelProto()

# Describe how to run the model
art = popart.AnchorReturnType("ALL")
dataFlow = popart.DataFlow(1, {o: art, w: art})

# Create a session to compile and execute the graph
trainingOptions = popart.SessionOptions()
session = popart.TrainingSession(
    fnModel=proto,
    dataFeed=dataFlow,
    losses=[popart.NllLoss(o, lb, "loss")],
    optimizer=popart.ConstSGD(0.001),
    userOptions=trainingOptions,
    deviceInfo=popart.DeviceManager().createIpuModelDevice({}))

# Compile graph
session.prepareDevice()

# Create buffers to receive results from the execution
anchors = session.initAnchorArrays()

# Generate some random input data
data_a = np.random.rand(1, 2).astype(np.float16)
data_lb = np.random.rand(1).astype(np.int32)

# Copy the weights to the device from the host
session.weightsFromHost()

stepio = popart.PyStepIO({a: data_a, lb: data_lb}, anchors)
for _ in range(6):
    session.run(stepio)

# Copy the weights to the host from the device
session.weightsToHost()

print("Input a is " + str(data_a))
print("Weight w is " + str(anchors[w]))
print("Result is " + str(anchors[o]))
