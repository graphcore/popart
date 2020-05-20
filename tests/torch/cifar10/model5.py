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
batchSize = 3
batchesPerStep = 2
anchors = {
    "nllLossVal": popart.AnchorReturnType("Final"),
}
dataFlow = popart.DataFlow(batchesPerStep, anchors)
inputShapeInfo = popart.InputShapeInfo()
inputShapeInfo.add("image0",
                   popart.TensorInfo("FLOAT", [batchSize, nInChans, 32, 32]))
inputShapeInfo.add("label", popart.TensorInfo("INT32", [batchSize]))
inNames = ["image0", "label"]
cifarInIndices = {"image0": 0, "label": 1}
outNames = ["nllLossVal"]


def nllloss(logprobs, targets):
    targets = targets.unsqueeze(1)
    loss = torch.gather(logprobs, 1, targets)
    return -loss.sum()


class Module0(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.conv1 = torchwriter.conv3x3(nInChans, nOutChans)
        # for softmax dim -1 is correct for [sample][class],
        # gives class probabilities for each sample.
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, inputs):
        image0 = inputs[0]
        x = self.conv1(image0)
        window_size = (int(x.size()[2]), int(x.size()[3]))
        x = torch.nn.functional.avg_pool2d(x, kernel_size=window_size)
        preprobs = torch.squeeze(x)
        probs = self.softmax(preprobs)
        logprobs = torch.log(probs)
        loss = nllloss(logprobs, inputs[1])
        return loss


# Set arbitrary seed so model weights are initialized to the
# same values each time the test is run
torch.manual_seed(1)

torchWriter = torchwriter.PytorchNetWriter(
    inNames=inNames,
    outNames=outNames,
    # large weight_decay term to test that it is definitely working
    optimizer=popart.SGD({
        "defaultLearningRate": (0.001, False),
        "defaultWeightDecay": (10.0, False)
    }),
    inputShapeInfo=inputShapeInfo,
    dataFlow=dataFlow,
    ### Torch specific:
    module=Module0(),
    samplesPerBatch=batchSize)

c10driver.run(torchWriter, None, args.outputdir, cifarInIndices, args.device,
              args.hw_id)
