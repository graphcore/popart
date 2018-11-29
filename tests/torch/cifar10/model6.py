# see model0.py for a more detailed
# description of what's going on.

import sys
import os
import torch
import c10driver
import cmdline
import poponnx
import poponnx_core
from poponnx.torch import torchwriter

args = cmdline.parse()

nInChans = 3
nOutChans = 10
batchSize = 2
batchesPerStep = 3
anchors = ["nllLossVal", "l1LossVal", "probs"]
art = poponnx_core.AnchorReturnType.ALL
dataFeed = poponnx_core.DataFlow(batchesPerStep, batchSize, anchors, art)
earlyInfo = poponnx_core.EarlyInfo()
earlyInfo.add("image0",
              poponnx_core.TensorInfo("FLOAT", [batchSize, nInChans, 32, 32]))
earlyInfo.add("image1",
              poponnx_core.TensorInfo("FLOAT", [batchSize, nInChans, 32, 32]))
earlyInfo.add("label", poponnx_core.TensorInfo("INT32", [batchSize]))
inNames = ["image0", "image1"]
cifarInIndices = {"image0": 0, "image1": 0, "label": 1}
outNames = ["preProbSquared", "probs"]
losses = [
    poponnx_core.NllLoss("probs", "label", "nllLossVal"),
    poponnx_core.L1Loss("preProbSquared", "l1LossVal", 0.01)
]

willowOptPasses = poponnx.Patterns()
willowOptPasses.OpToIdentity = True


class Module0(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.conv1 = torchwriter.conv3x3(nInChans, nOutChans)
        self.conv2 = torchwriter.conv3x3(nOutChans, nOutChans)
        self.conv3 = torchwriter.conv3x3(nOutChans, nOutChans)
        self.relu = torch.nn.functional.relu
        # for softmax dim -1 is correct for [sample][class],
        # gives class probabilities for each sample.
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, inputs):
        image0 = inputs[0]
        image1 = inputs[1]
        x = image0 - image1

        x = self.conv1(x)
        x = self.relu(x)
        # interestingly if this is self.conv2,
        # the weights are the same (shared)
        x = self.conv3(x)
        preProbSquared = x + x

        window_size = (int(x.size()[2]), int(x.size()[3]))
        x = torch.nn.functional.avg_pool2d(x, kernel_size=window_size)
        x = torch.squeeze(x)
        # probabilities:
        # Note that for Nll, Pytorch requires logsoftmax input.
        # We do this separately the framework dependant section,
        # torchwriter.py
        probs = self.softmax(x)
        # -> currently no support from pytorch
        # -> for gather or log (pytorch 0.4.1)
        # x = torch.gather(input = x, dim = 1, index= labels)
        # loss = torch.log(x)
        return preProbSquared, probs


torchWriter = torchwriter.PytorchNetWriter(
    inNames=inNames,
    outNames=outNames,
    losses=losses,
    optimizer=poponnx_core.ConstSGD(0.001),
    earlyInfo=earlyInfo,
    dataFeed=dataFeed,
    ### Torch specific:
    module=Module0())

c10driver.run(torchWriter, willowOptPasses, args.outputdir, cifarInIndices,
              args.device, args.hw_id)
