# Copyright (c) 2018 Graphcore Ltd. All rights reserved.
import sys
import os
import c10driver
import popart
import cmdline
from popart.torch import torchwriter
#we require torch in this file to create the torch Module
import torch

args = cmdline.parse()

nChans = 3

# process batchSize = 2 samples at a time,
# so weights updated on average gradient of
# batchSize = 2 samples. batchSize
# is EXACTLY the batch size.
batchSize = 2

# Return requested tensors every batchesPerStep = 4 cycles.
# so only communicate back to host every 2*4 = 8 samples.
batchesPerStep = 4

# Anchors, and how they are returned.
# Last batch in step, all samples in step,
# every N batches in a step? See ir.hpp for details.
# In this example:
# the l1 loss "out",
# and the input tensor "image0"
anchors = {
    "out": popart.AnchorReturnType("Final"),
    "image0": popart.AnchorReturnType("All")
}

dataFlow = popart.DataFlow(batchesPerStep, anchors)

# willow is non-dynamic. All input Tensor shapes and
# types must be fed into the Session constructor.
# In this example there is 1 streamed input, image0.
inputShapeInfo = popart.InputShapeInfo()
inputShapeInfo.add("image0",
                   popart.TensorInfo("FLOAT", [batchSize, nChans, 32, 32]))

inNames = ["image0"]
outNames = ["out"]

#cifar training data loader : at index 0 : image, at index 1 : label.
cifarInIndices = {"image0": 0}


class Module0(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.conv1 = torchwriter.conv3x3(nChans, nChans)
        self.conv2 = torchwriter.conv3x3(nChans, nChans)
        self.relu = torch.nn.functional.relu

    def forward(self, inputs):
        image0 = inputs[0]
        x = self.conv1(image0)
        y = self.conv2(image0)
        x = torch.cat([x, y])
        x = self.relu(x)
        x = torch.sum(0.1 * torch.abs(x))
        return x


# Set arbitrary seed so model weights are initialized to the
# same values each time the test is run
torch.manual_seed(2)

torchWriter = torchwriter.PytorchNetWriter(
    inNames=inNames,
    outNames=outNames,
    optimizer=popart.ConstSGD(0.001),
    inputShapeInfo=inputShapeInfo,
    dataFlow=dataFlow,
    ### Torch specific:
    module=Module0(),
    samplesPerBatch=batchSize)

willowOptPasses = None
c10driver.run(torchWriter, willowOptPasses, args.outputdir, cifarInIndices,
              args.device, args.hw_id)
