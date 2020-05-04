# Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#This test is not live until T7159 has been resolved

import sys
import os
import c10driver
import popart
import cmdline
from popart.torch import torchwriter
import torch
args = cmdline.parse()

nInChans = 3
nOutChans = 8
batchSize = 2
batchesPerStep = 4
anchors = {
    "l1LossVal": popart.AnchorReturnType("EveryN", 2),
    "out": popart.AnchorReturnType("Final"),
    "im0": popart.AnchorReturnType("All")
}
dataFeed = popart.DataFlow(batchesPerStep, anchors)
inputShapeInfo = popart.InputShapeInfo()
inputShapeInfo.add("im0",
                   popart.TensorInfo("FLOAT", [batchSize, nInChans, 32, 32]))

inNames = ["im0"]
outNames = ["out"]
cifarInIndices = {"im0": 0}
losses = [popart.L1Loss("out", "l1LossVal", 0.1)]


class Module0(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)

        self.sin = torch.sin
        self.conv1 = torchwriter.conv3x3(nInChans, nOutChans)
        self.conv2 = torchwriter.conv3x3(nOutChans, nOutChans)
        self.bn2 = torch.nn.BatchNorm2d(nOutChans, eps=0.1)
        self.conv3 = torchwriter.conv3x3(nOutChans, nOutChans)
        self.bn3 = torch.nn.BatchNorm2d(nOutChans, eps=0.1)

    def forward(self, inputs):
        im0 = inputs[0]
        x = self.conv1(im0)
        x = self.sin(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.sin(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.sin(x)
        return x


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
    module=Module0(),
    samplesPerBatch=batchSize)

c10driver.run(torchWriter,
              None,
              args.outputdir,
              cifarInIndices,
              args.device,
              args.hw_id,
              transformations=["prepareNodesForTraining"])
#, "removeUnusedInputs"])
