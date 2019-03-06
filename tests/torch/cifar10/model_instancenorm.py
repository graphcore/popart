import sys
import os
import c10driver
import poponnx
import cmdline
from poponnx.torch import torchwriter
import torch
args = cmdline.parse()

nInChans = 3
nOutChans = 8
batchSize = 2
batchesPerStep = 4
anchors = {
    "l1LossVal": poponnx.AnchorReturnType("EVERYN", 2),
    "out": poponnx.AnchorReturnType("FINAL"),
    "im0": poponnx.AnchorReturnType("ALL")
}
dataFeed = poponnx.DataFlow(batchesPerStep, anchors)
inputShapeInfo = poponnx.InputShapeInfo()
inputShapeInfo.add("im0",
                   poponnx.TensorInfo("FLOAT", [batchSize, nInChans, 32, 32]))

inNames = ["im0"]
outNames = ["out"]
cifarInIndices = {"im0": 0}
losses = [poponnx.L1Loss("out", "l1LossVal", 0.1)]


class Module0(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)

        self.sin = torch.sin
        self.conv1 = torchwriter.conv3x3(nInChans, nOutChans)
        self.conv2 = torchwriter.conv3x3(nOutChans, nOutChans)
        self.in2 = torch.nn.InstanceNorm2d(
            nOutChans, eps=0.1, affine=True, momentum=0)
        self.conv3 = torchwriter.conv3x3(nOutChans, nOutChans)
        self.in3 = torch.nn.InstanceNorm2d(
            nOutChans, eps=0.1, affine=True, momentum=0)

    def forward(self, inputs):
        im0 = inputs[0]
        x = self.conv1(im0)
        x = self.sin(x)
        x = self.conv2(x)
        x = self.in2(x)
        x = self.sin(x)
        x = self.conv3(x)
        x = self.in3(x)
        x = self.sin(x)
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

c10driver.run(
    torchWriter,
    None,
    args.outputdir,
    cifarInIndices,
    args.device,
    args.hw_id,
    transformations=["prepareNodesForTraining"])
