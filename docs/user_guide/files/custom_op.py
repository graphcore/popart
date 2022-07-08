# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import ctypes
import os

import numpy as np
import popart
import torch
import torch.nn as nn

# Load the .so file in the same directory.
myso = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "custom_op/build/custom_op.so"
)
myop = ctypes.cdll.LoadLibrary(myso)

batch_size = 1
chans_in = 2
chans_out = 3
size = 4
kernel_size = 3
padding = 1

# Create a builder
builder = popart.Builder()

# Assuming a batch size of 1, the inputs are:
x = builder.addInputTensor(
    popart.TensorInfo("FLOAT", [batch_size, chans_in, size, size])
)
# Let's assume that the number of output channels for the 1st convolution is 32
# and kernel size is 8
weights = np.random.rand(chans_out, chans_in, kernel_size, kernel_size).astype(
    np.float32
)
w = builder.addInitializedInputTensor(weights, "weights/1")

# Modify the input with a custom op that produces an output with the
# same shape as the input
leaky_relu = builder.customOp(
    opName="LeakyRelu", opVersion=1, domain="custom.ops", inputs=[x], attributes={}
)[0]

# This shows the ONNX shape inference is working.
print("Shape of {}: {}".format(leaky_relu, builder.getTensorShape(leaky_relu)))

# Let's create a known tensor, but with an undefined shape
y = builder.aiOnnx.conv(
    [leaky_relu, w], dilations=[1, 1], pads=[padding] * 4, strides=[1, 1]
)

l1 = builder.aiGraphcore.l1loss([y], 1.0)

proto = builder.getModelProto()

art = popart.AnchorReturnType("All")
# Describe how to run the model
dataflow = popart.DataFlow(1, {y: art, leaky_relu: art, w: art, l1: art})

# Create a session to compile and execute the graph
options = popart.SessionOptions()
device = popart.DeviceManager().createIpuModelDevice({})
session = popart.TrainingSession(
    fnModel=proto,
    dataFlow=dataflow,
    loss=l1,
    optimizer=popart.ConstSGD(0.001),
    userOptions=options,
    deviceInfo=device,
)

# Compile graph
session.prepareDevice()
# Create buffers to receive results from the execution
anchors = session.initAnchorArrays()
# Copy weights onto the IPU
session.weightsFromHost()
# Generate some random input data.
myinput = 10 * np.random.rand(batch_size, chans_in, size, size).astype(np.float32)
print("Input is : {}", myinput)

stepio = popart.PyStepIO({x: myinput}, anchors)
session.run(stepio)
print("Weights are: {}", anchors[w])
print("Output of leaky_relu op is: {}", anchors[leaky_relu])


# Check we have got it right with a torch comparison.
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(chans_in, chans_out, kernel_size, padding=[padding])
        self.conv.weight.data = torch.tensor(weights)
        self.conv.bias.data = torch.tensor([0.0 for i in range(chans_out)])
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, input_):
        input_ = self.conv(input_)
        input_ = self.leaky_relu(input_)
        return input_


net = Net()

optimizer = torch.optim.SGD(net.parameters(), lr=0.001)
loss = nn.L1Loss()
optimizer.zero_grad()
result = net(torch.tensor(myinput))
output = loss(result, torch.zeros(result.shape))
output.backward()

optimizer.step()

# Check the output of the leaky_relu op
assert np.allclose(anchors[l1], output.detach().numpy())
assert np.allclose(anchors[y], result.detach().numpy())
print(f"Result is {str(anchors[y])}")
