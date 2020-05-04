# Copyright (c) 2018 Graphcore Ltd. All rights reserved.
# see model0.py for a more detailed
# description of what's going on.

import sys
import os

import c10driver
import popart
import cmdline
from popart.torch import torchwriter
#we require torch in this file to create the torch Module
import torch

args = cmdline.parse()

nInChans = 3
nOutChans = 10
batchSize = 2
batchesPerStep = 3
anchors = {"l1LossVal": popart.AnchorReturnType("Final")}
dataFeed = popart.DataFlow(batchesPerStep, anchors)
inputShapeInfo = popart.InputShapeInfo()
inputShapeInfo.add("image0",
                   popart.TensorInfo("FLOAT", [batchSize, nInChans, 32, 32]))
inputShapeInfo.add("image1",
                   popart.TensorInfo("FLOAT", [batchSize, nInChans, 32, 32]))
inputShapeInfo.add("label", popart.TensorInfo("INT32", [batchSize]))
inNames = ["image0", "image1"]
cifarInIndices = {"image0": 0, "image1": 0, "label": 1}
outNames = ["out"]
losses = [popart.L1Loss("out", "l1LossVal", 0.01)]

willowOptPasses = popart.Patterns()
willowOptPasses.SubtractArg1GradOp = True


class Module0(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.conv1 = torchwriter.conv3x3(nInChans, nOutChans)
        self.conv2 = torchwriter.conv3x3(nInChans, nOutChans)

    def forward(self, inputs):
        x = self.conv1(inputs[0])
        y = self.conv2(inputs[1])
        y = torch.sum(y, dim=(0, 1))
        x = x - y
        y = torch.sum(y, dim=0)
        return x + y


# Set arbitrary seed so model weights are initialized to the
# same values each time the test is run
torch.manual_seed(1)

torchWriter = torchwriter.PytorchNetWriter(
    inNames=inNames,
    outNames=outNames,
    losses=losses,
    optimizer=popart.ConstSGD(0.001),
    inputShapeInfo=inputShapeInfo,
    dataFeed=dataFeed,
    ### Torch specific:
    module=Module0())

c10driver.run(torchWriter, willowOptPasses, args.outputdir, cifarInIndices,
              args.device, args.device_hw_id)
