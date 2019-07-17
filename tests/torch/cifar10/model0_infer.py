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
batchesPerStep = 4
anchors = {"out": poponnx.AnchorReturnType("EVERYN", 2)}
dataFeed = poponnx.DataFlow(batchesPerStep, anchors)
inputShapeInfo = poponnx.InputShapeInfo()
samplesPerBatch = 6
inputShapeInfo.add(
    "image0", poponnx.TensorInfo("FLOAT", [samplesPerBatch, nChans, 32, 32]))

inNames = ["image0"]
outNames = ["out"]
losses = []
optimizer = None

#cifar training data loader : at index 0 : image, at index 1 : label.
cifarInIndices = {"image0": 0}


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
    optimizer=optimizer,
    inputShapeInfo=inputShapeInfo,
    dataFeed=dataFeed,
    ### Torch specific:
    module=Module0(),
    samplesPerBatch=samplesPerBatch)

# Passes if torch and poponnx models match
c10driver.run(torchWriter=torchWriter,
              passes=None,
              outputdir=args.outputdir,
              cifarInIndices=cifarInIndices,
              device=args.device,
              device_hw_id=args.hw_id,
              mode="infer")
