# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
import ctypes
import os

# Load the .so file in the same directory.
myso = os.path.join(os.path.dirname(os.path.realpath(__file__)), "cube.so")
myop = ctypes.cdll.LoadLibrary(myso)

# Create a builder
builder = popart.Builder()

# Assuming a batch size of 1, the inputs are:
x = builder.addInputTensor(popart.TensorInfo("FLOAT", [1, 4, 84, 84]))
# Let's assume that the number of output channels for the 1st convolution is 32
# and kernel size is 8
weights = np.random.rand(32, 4, 8, 8).astype(np.float32)
w = builder.addInitializedInputTensor(weights, "weights/1")

# Modify the input with a custom op that produces an output with the
# same shape as the input
cube = builder.customOp(opName="Cube",
                        opVersion=1,
                        domain="ai.acme",
                        inputs=[x],
                        attributes={})[0]

# This shows the ONNX shape inference is working.
print("Shape of {}: {}".format(cube, builder.getTensorShape(cube)))

# Let's create a known tensor, but with an undefined shape
y = builder.aiOnnx.conv([cube, w],
                        dilations=[1, 1],
                        pads=[1, 1, 1, 1],
                        strides=[1, 1])

builder.addOutputTensor(y)

proto = builder.getModelProto()

art = popart.AnchorReturnType("All")
# Describe how to run the model
dataflow = popart.DataFlow(1, {y: art, cube: art, w: art})

# Create a session to compile and execute the graph
options = popart.SessionOptions()
device = popart.DeviceManager().createIpuModelDevice({})
session = popart.TrainingSession(fnModel=proto,
                                 dataFeed=dataflow,
                                 losses=[popart.L1Loss(y, "l1LossVal", 0.1)],
                                 optimizer=popart.ConstSGD(0.001),
                                 userOptions=options,
                                 deviceInfo=device)

# Compile graph
session.prepareDevice()
# Create buffers to receive results from the execution
anchors = session.initAnchorArrays()
# Copy weights onto the IPU
session.weightsFromHost()
# Generate some random input data. Careful, cube will create some large
# outputs, watch for overflows!
myinput = 10 * np.random.rand(1, 4, 84, 84).astype(np.float32)
print("Input is : {}", myinput)

stepio = popart.PyStepIO({x: myinput}, anchors)
session.run(stepio)
print("Weights are: {}", anchors[w])
print("Output of cube op is: {}", anchors[cube])
# Check the output of the cube op
assert np.allclose(anchors[cube], (myinput**3))
print("Result is {}" + str(anchors[y]))
