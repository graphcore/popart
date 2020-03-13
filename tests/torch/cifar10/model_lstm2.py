# Copyright (c) 2018 Graphcore Ltd. All rights reserved.
import sys
import os
import c10driver
import popart
import cmdline
from popart.torch import torchwriter
#we require torch in this file to create the torch Module
import torch
import numpy as np

args = cmdline.parse()

nChans = 3

# process batchSize = 2 samples at a time,
# so weights updated on average gradient of
# batchSize = 2 samples. batchSize
# is EXACTLY the batch size.
batchSize = 2

# Return requested tensors every batchesPerStep = 4 cycles.
# so only communicate back to host every 2*4 = 8 samples.
batchesPerStep = 4

# Anchors, and how they are returned.
# Last batch in step, all samples in step,
# every N batches in a step? See ir.hpp for details.
# In this example:
# the l1 loss "l1LossVal",
# the tensor to which the loss is applied "out",
# and the input tensor "image0"
anchors = {
    "l1LossVal": popart.AnchorReturnType("EVERYN", 2),
    "out": popart.AnchorReturnType("FINAL"),
    "image0": popart.AnchorReturnType("ALL")
}

dataFeed = popart.DataFlow(batchesPerStep, anchors)

# willow is non-dynamic. All input Tensor shapes and
# types must be fed into the Session constructor.
# In this example there is 1 streamed input, image0.
inputShapeInfo = popart.InputShapeInfo()
inputShapeInfo.add("image0",
                   popart.TensorInfo("FLOAT", [batchSize, nChans, 32, 32]))

inNames = ["image0"]

# outNames: not the same as anchors,
# these are the Tensors which will be
# connected to the loss layers
outNames = ["out"]

#cifar training data loader : at index 0 : image, at index 1 : label.
cifarInIndices = {"image0": 0}

losses = [popart.L1Loss("out", "l1LossVal", 0.1)]

layers = 1
batch_size = 32
hidden_size = 1


class Module0(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.lstm = torch.nn.LSTM(4, 4, 1)

    def forward(self, inputs):
        """out = relu(conv(in))"""
        image0 = inputs[0]
        image0 = image0[0:1]
        i = image0[:, :, 0:1, 0:4]

        # seq_length 3
        # batch_size 1
        # input_size 4
        a = i[0:1].view(3, 1, 4)
        b = i[0:1].view(3, 1, 4)

        h0 = b[0:1, 0:4].view(1, 1, 4)
        c0 = b[1:2, 0:4].view(1, 1, 4)
        x = self.lstm(a, (h0, c0))
        result = x[0]
        return result


# Set arbitrary seed so model weights are initialized to the
# same values each time the test is run
torch.manual_seed(1)

torchWriter = torchwriter.PytorchNetWriter(
    inNames=inNames,
    outNames=outNames,
    losses=losses,
    optimizer=popart.ConstSGD(0.001),
    inputShapeInfo=inputShapeInfo,
    dataFeed=dataFeed,
    ### Torch specific:
    module=Module0(),
    samplesPerBatch=batchSize)

c10driver.run(torchWriter, popart.Patterns(['PreUniRepl']), args.outputdir,
              cifarInIndices, args.device, args.hw_id)
