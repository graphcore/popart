# see model0.py for a more detailed
# description of what's going on.

import sys
import os
import torch
import numpy as np
from torchvision import transforms, datasets
import c10driver
import poponnx
from poponnx.torch import torchwriter

if (len(sys.argv) != 2):
    raise RuntimeError("onnx_net.py <log directory>")

outputdir = sys.argv[1]
if not os.path.exists(outputdir):
    print("Making %s" % (outputdir, ))
    os.mkdir(outputdir)

nInChans = 3
nOutChans = 10
samplesPerBatch = 2
batchesPerStep = 3
anchors = ["l1LossVal", "out"]
art = poponnx.AnchorReturnType.ALL
dataFeed = poponnx.DataFlow(batchesPerStep, samplesPerBatch, anchors, art)
earlyInfo = poponnx.EarlyInfo()
earlyInfo.add("image0",
              poponnx.TensorInfo("FLOAT", [samplesPerBatch, nInChans, 32, 32]))
inNames = ["image0"]
cifarInIndices = {"image0": 0, "label": 1}
outNames = ["out"]
losses = [poponnx.L1Loss("out", "l1LossVal", 0.1)]
willowOptPasses = ["PreUniRepl", "PostNRepl", "SoftmaxGradDirect"]


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
    earlyInfo=earlyInfo,
    dataFeed=dataFeed,
    ### Torch specific:
    module=Module0())

c10driver.run(torchWriter, willowOptPasses, outputdir, cifarInIndices)
