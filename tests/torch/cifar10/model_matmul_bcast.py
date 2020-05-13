# Copyright (c) 2018 Graphcore Ltd. All rights reserved.
import c10driver

import sys
import os
import popart
import popart_core
import cmdline
from popart.torch import torchwriter
import torch

args = cmdline.parse()

nChans = 3
oChans = 10
# process batchSize = 2 samples at a time,
# so weights updated on average gradient of
# batchSize = 2 samples. batchSize
# is EXACTLY the batch size.
batchSize = 2

# Return requested tensors every batchesPerStep = 3 cycles.
# so only communicate back to host every 2*3 = 6 samples.
batchesPerStep = 3

# anchors, and how they are returned: in this example,
# return the l1 loss "l1LossVal",
anchors = {
    "l1LossVal": popart_core.AnchorReturnType("Final"),
}
dataFeed = popart_core.DataFlow(batchesPerStep, anchors)

# willow is non-dynamic. All input Tensor shapes and
# types must be fed into the WillowNet constructor.
# In this example there is 1 streamed input, image0.
inputShapeInfo = popart_core.InputShapeInfo()
inputShapeInfo.add("image0",
                   popart.TensorInfo("FLOAT", [batchSize, nChans, 32, 32]))

inNames = ["image0"]
outNames = ["l1LossVal"]

#cifar training data loader : at index 0 : image, at index 1 : label.
cifarInIndices = {"image0": 0}  # not used: "label": 1}

# The optimization passes to run in the Ir, see patterns.hpp
willowOptPasses = popart.Patterns()
willowOptPasses.OpToIdentity = True


class Module0(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.conv1 = torchwriter.conv3x3(nChans, oChans)
        self.conv2 = torchwriter.conv3x3(oChans, oChans)
        self.weights = torch.nn.Parameter(torch.rand(3, 10, 5))

        self.relu = torch.nn.functional.relu
        self.matmul = torch.matmul

    def forward(self, inputs):
        """out = relu(matmul(in, weights))"""
        image0 = inputs[0]
        x = self.relu(image0)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        window_size = (int(x.size()[2]), int(x.size()[3]))
        x = torch.nn.functional.avg_pool2d(x, kernel_size=window_size)
        x = torch.squeeze(x)

        weights = self.weights
        x = self.matmul(x, weights)
        out = torch.sum(0.1 * torch.abs(x))  # l1loss

        return out


# Set arbitrary seed so model weights are initialized to the
# same values each time the test is run
torch.manual_seed(1)

torchWriter = torchwriter.PytorchNetWriter(
    inNames=inNames,
    outNames=outNames,
    optimizer=popart_core.ConstSGD(0.001),
    inputShapeInfo=inputShapeInfo,
    dataFeed=dataFeed,
    ### Torch specific:
    module=Module0(),
    samplesPerBatch=batchSize)

c10driver.run(torchWriter, willowOptPasses, args.outputdir, cifarInIndices,
              args.device, args.hw_id)
