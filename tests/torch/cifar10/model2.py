# Copyright (c) 2018 Graphcore Ltd. All rights reserved.
# see model0.py for a more detailed
# description of what's going on.

import sys
import os
import torch
import numpy as np
from torchvision import transforms, datasets
import c10driver
import popart
import cmdline
from popart.torch import torchwriter

args = cmdline.parse()

nInChans = 3
nOutChans = 10
batchSize = 2
batchesPerStep = 3
anchors = {
    "l1LossVal": popart.AnchorReturnType("FINAL"),
    "out": popart.AnchorReturnType("FINAL")
}
dataFeed = popart.DataFlow(batchesPerStep, anchors)
inputShapeInfo = popart.InputShapeInfo()
inputShapeInfo.add("image0",
                   popart.TensorInfo("FLOAT", [batchSize, nInChans, 32, 32]))
inNames = ["image0"]
cifarInIndices = {"image0": 0}
outNames = ["out"]
losses = [popart.L1Loss("out", "l1LossVal", 0.1)]


class Module0(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.conv1 = torchwriter.conv3x3(nInChans, nOutChans)
        self.conv2 = torchwriter.conv3x3(nOutChans, nOutChans)
        self.sin = torch.sin
        # number of classes = nOutChans
        self.linear = x = torch.nn.Linear(nOutChans, nOutChans)

    def forward(self, inputs):
        image0 = inputs[0]
        x = self.sin(image0)
        x = self.conv1(x)
        x = self.sin(x)
        x = self.conv2(x)
        window_size = (int(x.size()[2]), int(x.size()[3]))
        x = torch.nn.functional.avg_pool2d(x, kernel_size=window_size)
        x = torch.squeeze(x)
        # This is the where the GEMM happens:
        out = self.linear(x)
        return out


# Set arbitrary seed so model weights are initialized to the
# same values each time the test is run
torch.manual_seed(1)

torchWriter = torchwriter.PytorchNetWriter(
    inNames=inNames,
    outNames=outNames,
    losses=losses,
    # large weight_decay term to test that it is definitely working
    optimizer=popart.ConstSGD(learning_rate=0.001, weight_decay=10),
    inputShapeInfo=inputShapeInfo,
    dataFeed=dataFeed,
    ### Torch specific:
    module=Module0(),
    samplesPerBatch=batchSize)

c10driver.run(torchWriter, None, args.outputdir, cifarInIndices, args.device,
              args.hw_id)
