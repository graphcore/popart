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

nChans = 3
samplesPerBatch = 2
batchesPerStep = 3
anchors = ["l1LossVal", "out", "image0"]
art = poponnx.AnchorReturnType.ALL
dataFeed = poponnx.DataFlow(batchesPerStep, samplesPerBatch, anchors, art)
earlyInfo = poponnx.EarlyInfo()
earlyInfo.add("image0",
              poponnx.TensorInfo("FLOAT", [samplesPerBatch, nChans, 32, 32]))

inNames = ["image0"]
outNames = ["out"]
losses = [poponnx.L1Loss("out", "l1LossVal", 0.1)]

#cifar training data loader : at index 0 : image, at index 1 : label.
cifarInIndices = {"image0": 0}

# The optimization passes to run in the Ir, see patterns.hpp
willowOptPasses = ["PreUniRepl", "PostNRepl", "SoftmaxGradDirect"]


class Module0(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.conv1 = torch.nn.Conv2d(
            nChans,
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
    earlyInfo=earlyInfo,
    dataFeed=dataFeed,
    ### Torch specific:
    module=Module0())

# Passes if torch and poponnx models match
c10driver.run(
    torchWriter=torchWriter,
    passes=willowOptPasses,
    outputdir=outputdir,
    cifarInIndices=cifarInIndices,
    mode="evaluate")
