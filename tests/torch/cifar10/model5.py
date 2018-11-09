# see model0.py for a more detailed
# description of what's going on.

#import c10 driver first, as it appends the necessary
# paths to sys.path. This is a temporary solution
import c10driver

import sys
import os
import poponnx_core
from poponnx.torch import torchwriter
#we require torch in this file to create the torch Module
import torch

if (len(sys.argv) != 2):
    raise RuntimeError("onnx_net.py <log directory>")

outputdir = sys.argv[1]
if not os.path.exists(outputdir):
    print("Making %s" % (outputdir, ))
    os.mkdir(outputdir)

nInChans = 3
nOutChans = 10
samplesPerBatch = 3
batchesPerStep = 2
anchors = ["nllLossVal", "probs"]
art = poponnx_core.AnchorReturnType.ALL
dataFeed = poponnx_core.DataFlow(batchesPerStep, samplesPerBatch, anchors, art)
earlyInfo = poponnx_core.EarlyInfo()
earlyInfo.add(
    "image0",
    poponnx_core.TensorInfo("FLOAT", [samplesPerBatch, nInChans, 32, 32]))
earlyInfo.add("label", poponnx_core.TensorInfo("INT32", [samplesPerBatch]))
inNames = ["image0"]
cifarInIndices = {"image0": 0, "label": 1}
outNames = ["probs"]
losses = [poponnx_core.NllLoss("probs", "label", "nllLossVal")]
willowOptPasses = ["PreUniRepl", "PostNRepl", "SoftmaxGradDirect"]


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
    optimizer=poponnx_core.ConstSGD(0.001),
    earlyInfo=earlyInfo,
    dataFeed=dataFeed,
    ### Torch specific:
    module=Module0())

c10driver.run(torchWriter, willowOptPasses, outputdir, cifarInIndices)
