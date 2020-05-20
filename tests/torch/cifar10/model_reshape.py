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

# the number of channels of a cifar-10 image
nInChans = 3
# this should be even for this experiment,
# as it is divided by 2 in the reshape
nOutChans = 10
batchSize = 2
batchesPerStep = 3
anchors = {
    "l1LossVal": popart.AnchorReturnType("All"),
}

dataFlow = popart.DataFlow(batchesPerStep, anchors)
earlyInfo = popart.InputShapeInfo()
earlyInfo.add("image0",
              popart.TensorInfo("FLOAT", [batchSize, nInChans, 32, 32]))
inNames = ["image0"]
cifarInIndices = {"image0": 0}
outNames = ["l1LossVal"]
willowOptPasses = [
    "PreUniRepl", "PostNRepl", "SoftmaxGradDirect", "OpToIdentity", "Inplace0"
]


class Module0(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.conv1 = torchwriter.conv3x3(nInChans, nOutChans)
        self.relu = torch.nn.functional.relu

    def forward(self, inputs):
        image0 = inputs[0]
        x = image0
        x = self.conv1(x)
        x = self.relu(x)

        nSamples = int(x.size()[0])
        nChans = int(x.size()[1])
        window_size = (int(x.size()[2]), int(x.size()[3]))
        x = torch.nn.functional.avg_pool2d(x, kernel_size=window_size)
        # squeeze out the 1s in the shape
        x = torch.squeeze(x)
        # this introduces a Reshape:
        x = x.view(nSamples, nChans // 2, 2)
        # this would add a Flatten:
        # x = x.view(nSamples, nChans)
        preProbSquared = x + x
        l1loss = torch.sum(0.1 * torch.abs(preProbSquared))
        return l1loss


# Set arbitrary seed so model weights are initialized to the
# same values each time the test is run
torch.manual_seed(1)

torchWriter = torchwriter.PytorchNetWriter(
    inNames=inNames,
    outNames=outNames,
    optimizer=popart.ConstSGD(0.001),
    inputShapeInfo=earlyInfo,
    dataFlow=dataFlow,
    ### Torch specific:
    module=Module0(),
    samplesPerBatch=batchSize)

c10driver.run(torchWriter, None, args.outputdir, cifarInIndices, args.device,
              args.hw_id)
