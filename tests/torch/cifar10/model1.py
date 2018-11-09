# see model0.py for a more detailed
# description of what's going on.

import sys
import os

import c10driver
import poponnx
from poponnx.torch import torchwriter
#we require torch in this file to create the torch Module
import torch

if (len(sys.argv) != 2):
    raise RuntimeError("onnx_net.py <log directory>")

outputdir = sys.argv[1]
if not os.path.exists(outputdir):
    print("Making %s" % (outputdir, ))
    os.mkdir(outputdir)

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
anchors = ["nllLossVal", "l1LossVal", "probs"]
art = poponnx.AnchorReturnType.ALL
dataFeed = poponnx.DataFlow(batchesPerStep, samplesPerBatch, anchors, art)
earlyInfo = poponnx.EarlyInfo()
earlyInfo.add(
    "image0",
    poponnx.TensorInfo("FLOAT", [samplesPerBatch, nInChans, 32, 32]))
earlyInfo.add(
    "image1",
    poponnx.TensorInfo("FLOAT", [samplesPerBatch, nInChans, 32, 32]))
earlyInfo.add("label", poponnx.TensorInfo("INT32", [samplesPerBatch]))
inNames = ["image0", "image1"]
cifarInIndices = {"image0": 0, "image1": 0, "label": 1}
outNames = ["preProbSquared", "probs"]
losses = [
    poponnx.NllLoss("probs", "label", "nllLossVal"),
    poponnx.L1Loss("preProbSquared", "l1LossVal", 0.01)
]
willowOptPasses = ["PreUniRepl", "PostNRepl", "SoftmaxGradDirect"]


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
        x = image0 + image1

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
    optimizer=poponnx.ConstSGD(0.001),
    earlyInfo=earlyInfo,
    dataFeed=dataFeed,
    ### Torch specific:
    module=Module0())

c10driver.run(torchWriter, willowOptPasses, outputdir, cifarInIndices)
