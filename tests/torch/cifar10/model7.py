import c10driver

import sys
import os
#import torch
import poponnx_core
from poponnx.torch import torchwriter
import torch

nChans = 3
oChans = 10
# process batchSize = 2 samples at a time,
# so weights updated on average gradient of
# batchSize = 2 samples. batchSize
# is EXACTLY the batch size.
batchSize = 2

# Return requested tensors every batchesPerStep = 3 cycles.
# so only communicate back to host every 2*3 = 6 samples.
batchesPerStep = 3

# anchors : in this example,
# return the l1 loss "l1LossVal",
# the tensor to which the loss is applied "out",
# and the input tensor "image0"
anchors = ["l1LossVal", "out"]

# What exactly should be returned of anchors?
# Last batch in step, all samples in step,
# sum over samples in step? See ir.hpp for details.
art = poponnx_core.AnchorReturnType.ALL

dataFeed = poponnx_core.DataFlow(batchesPerStep, batchSize, anchors, art)

# willow is non-dynamic. All input Tensor shapes and
# types must be fed into the WillowNet constructor.
# In this example there is 1 streamed input, image0.
earlyInfo = poponnx_core.EarlyInfo()
earlyInfo.add("image0",
              poponnx_core.TensorInfo("FLOAT", [batchSize, nChans, 32, 32]))

inNames = ["image0"]

# outNames: not the same as anchors,
# these are the Tensors which will be
# connected to the loss layers
outNames = ["out"]

#cifar training data loader : at index 0 : image, at index 1 : label.
cifarInIndices = {"image0": 0, "label": 1}

losses = [poponnx_core.L1Loss("out", "l1LossVal", 0.1)]

# The optimization passes to run in the Ir, see patterns.hpp
willowOptPasses = [
    "PreUniRepl", "PostNRepl", "SoftmaxGradDirect", "OpToIdentity"
]


class Module0(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.conv1 = torchwriter.conv3x3(nChans, oChans)
        self.conv2 = torchwriter.conv3x3(oChans, oChans)
        self.weights = torch.nn.Parameter(torch.rand(10, 5))

        self.relu = torch.nn.functional.relu
        self.matmul = torch.matmul

    def forward(self, inputs):
        """out = relu(matmul(in, weights))"""
        image0 = inputs[0]
        x = self.relu(image0)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        window_size = (int(x.size()[2]), int(x.size()[3]))
        x = torch.nn.functional.avg_pool2d(x, kernel_size=window_size)
        x = torch.squeeze(x)

        weights = self.weights
        out = self.matmul(x, weights)

        return out


torchWriter = torchwriter.PytorchNetWriter(
    inNames=inNames,
    outNames=outNames,
    losses=losses,
    optimizer=poponnx_core.ConstSGD(0.001),
    earlyInfo=earlyInfo,
    dataFeed=dataFeed,
    ### Torch specific:
    module=Module0())

try:
    outputdir = sys.argv[1]
except IndexError:
    outputdir = None

c10driver.run(torchWriter, willowOptPasses, outputdir, cifarInIndices)
