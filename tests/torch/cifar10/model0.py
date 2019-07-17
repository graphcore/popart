import sys
import os
import c10driver
import poponnx
import cmdline
from poponnx.torch import torchwriter
#we require torch in this file to create the torch Module
import torch

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
    "l1LossVal": poponnx.AnchorReturnType("EVERYN", 2),
    "out": poponnx.AnchorReturnType("FINAL"),
    "image0": poponnx.AnchorReturnType("ALL")
}

dataFeed = poponnx.DataFlow(batchesPerStep, anchors)

# willow is non-dynamic. All input Tensor shapes and
# types must be fed into the Session constructor.
# In this example there is 1 streamed input, image0.
inputShapeInfo = poponnx.InputShapeInfo()
inputShapeInfo.add("image0",
                   poponnx.TensorInfo("FLOAT", [batchSize, nChans, 32, 32]))

inNames = ["image0"]

# outNames: not the same as anchors,
# these are the Tensors which will be
# connected to the loss layers
outNames = ["out"]

#cifar training data loader : at index 0 : image, at index 1 : label.
cifarInIndices = {"image0": 0}

losses = [poponnx.L1Loss("out", "l1LossVal", 0.1)]


class Module0(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.conv1 = torch.nn.Conv2d(nChans,
                                     nChans,
                                     kernel_size=(3, 3),
                                     stride=1,
                                     padding=(1, 3),
                                     bias=False)
        self.relu = torch.nn.functional.relu

    def forward(self, inputs):
        """out = relu(conv(in))"""
        image0 = inputs[0]
        x = self.conv1(image0)
        x = self.relu(x)
        return x


# Set arbitrary seed so model weights are initialized to the
# same values each time the test is run
torch.manual_seed(1)

torchWriter = torchwriter.PytorchNetWriter(
    inNames=inNames,
    outNames=outNames,
    losses=losses,
    optimizer=poponnx.ConstSGD(0.001),
    inputShapeInfo=inputShapeInfo,
    dataFeed=dataFeed,
    ### Torch specific:
    module=Module0(),
    samplesPerBatch=batchSize)

c10driver.run(torchWriter, None, args.outputdir, cifarInIndices, args.device,
              args.hw_id)
