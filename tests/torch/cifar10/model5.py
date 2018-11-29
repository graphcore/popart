# see model0.py for a more detailed
# description of what's going on.

import sys
import os
import c10driver
import poponnx
import cmdline
from poponnx.torch import torchwriter
#we require torch in this file to create the torch Module
import torch

args = cmdline.parse()

nInChans = 3
nOutChans = 10
batchSize = 3
batchesPerStep = 2
anchors = ["nllLossVal", "probs"]
art = poponnx.AnchorReturnType.ALL
dataFeed = poponnx.DataFlow(batchesPerStep, batchSize, anchors, art)
earlyInfo = poponnx.EarlyInfo()
earlyInfo.add("image0",
              poponnx.TensorInfo("FLOAT", [batchSize, nInChans, 32, 32]))
earlyInfo.add("label", poponnx.TensorInfo("INT32", [batchSize]))
inNames = ["image0"]
cifarInIndices = {"image0": 0, "label": 1}
outNames = ["probs"]
losses = [poponnx.NllLoss("probs", "label", "nllLossVal")]


class Module0(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.conv1 = torchwriter.conv3x3(nInChans, nOutChans)
        self.relu = torch.nn.functional.relu
        # for softmax dim -1 is correct for [sample][class],
        # gives class probabilities for each sample.
        self.softmax = torch.nn.Softmax(dim=-1)
        # self.prebrobs = None

    def forward(self, inputs):
        image0 = inputs[0]
        x = self.conv1(image0)
        #  x = self.relu(x)
        window_size = (int(x.size()[2]), int(x.size()[3]))
        x = torch.nn.functional.avg_pool2d(x, kernel_size=window_size)
        preprobs = torch.squeeze(x)
        # probabilities:
        # Note that for Nll, Pytorch requires logsoftmax input.
        # We do this separately in the framework specfic code,
        # torchwriter.py
        probs = self.softmax(preprobs)
        # -> currently no support from pytorch
        # -> for gather or log (pytorch 0.4.1)
        # x = torch.gather(input = x, dim = 1, index= labels)
        # loss = torch.log(x)
        return probs


torchWriter = torchwriter.PytorchNetWriter(
    inNames=inNames,
    outNames=outNames,
    losses=losses,
    optimizer=poponnx.ConstSGD(0.001),
    earlyInfo=earlyInfo,
    dataFeed=dataFeed,
    ### Torch specific:
    module=Module0())

c10driver.run(torchWriter, None, args.outputdir, cifarInIndices, args.device,
              args.hw_id)
