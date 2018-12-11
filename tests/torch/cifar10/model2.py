# see model0.py for a more detailed
# description of what's going on.

import sys
import os
import torch
import numpy as np
from torchvision import transforms, datasets
import c10driver
import poponnx
import cmdline
from poponnx.torch import torchwriter

args = cmdline.parse()

nInChans = 3
nOutChans = 10
batchSize = 2
batchesPerStep = 3
anchors = {
    "l1LossVal": poponnx.AnchorReturnType("FINAL"),
    "out": poponnx.AnchorReturnType("FINAL")
}
dataFeed = poponnx.DataFlow(batchesPerStep, anchors)
inputShapeInfo = poponnx.InputShapeInfo()
inputShapeInfo.add("image0",
                   poponnx.TensorInfo("FLOAT", [batchSize, nInChans, 32, 32]))
inNames = ["image0"]
cifarInIndices = {"image0": 0, "label": 1}
outNames = ["out"]
losses = [poponnx.L1Loss("out", "l1LossVal", 0.1)]


class Module0(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.conv1 = torchwriter.conv3x3(nInChans, nOutChans)
        self.conv2 = torchwriter.conv3x3(nOutChans, nOutChans)
        self.relu = torch.nn.functional.relu
        # number of classes = nOutChans
        self.linear = x = torch.nn.Linear(nOutChans, nOutChans)

    def forward(self, inputs):
        image0 = inputs[0]
        x = self.relu(image0)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        window_size = (int(x.size()[2]), int(x.size()[3]))
        x = torch.nn.functional.avg_pool2d(x, kernel_size=window_size)
        x = torch.squeeze(x)
        # This is the where the GEMM happens:
        raise RuntimeError(
            "At this point, we're about to enter a linear layer, but poponnx hasn't implemented this yet"
        )
        out = self.linear(x)
        return out


torchWriter = torchwriter.PytorchNetWriter(
    inNames=inNames,
    outNames=outNames,
    losses=losses,
    optimizer=poponnx.ConstSGD(0.001),
    inputShapeInfo=inputShapeInfo,
    dataFeed=dataFeed,
    ### Torch specific:
    module=Module0())

c10driver.run(torchWriter, None, args.outputdir, cifarInIndices, args.device,
              args.hw_id)
