import sys
import os

# TODO this needs to be removed using __init__.py or some similar scheme
testdir = os.path.dirname(os.path.abspath(__file__))
libpath = os.path.join(testdir, "../../../lib")
sys.path.append(libpath)

import torch
import c10driver
import poponnx_core

from poponnx.torch import torchwriter

if (len(sys.argv) != 2):
    raise RuntimeError("onnx_net.py <log directory>")

outputdir = sys.argv[1]
if not os.path.exists(outputdir):
    print("Making %s" % (outputdir, ))
    os.mkdir(outputdir)

nChans = 3

# process samplesPerBatch = 2 samples at a time,
# so weights updated on average gradient of
# samplesPerBatch = 2 samples. samplesPerBatch
# is EXACTLY the batch size.
samplesPerBatch = 2

# Return requested tensors every batchesPerStep = 3 cycles.
# so only communicate back to host every 2*3 = 6 samples.
batchesPerStep = 3

# anchors : in this example,
# return the l1 loss "l1LossVal",
# the tensor to which the loss is applied "out",
# and the input tensor "image0"
anchors = ["l1LossVal", "out", "image0"]

# What exactly should be returned of anchors?
# Last batch in step, all samples in step,
# sum over samples in step? See ir.hpp for details.
art = poponnx_core.AnchorReturnType.ALL

dataFeed = poponnx_core.DataFlow(batchesPerStep, samplesPerBatch, anchors, art)

# willow is non-dynamic. All input Tensor shapes and
# types must be fed into the WillowNet constructor.
# In this example there is 1 streamed input, image0.
earlyInfo = poponnx_core.EarlyInfo()
earlyInfo.add("image0",
              poponnx_core.TensorInfo("FLOAT", [samplesPerBatch, nChans, 32, 32]))

inNames = ["image0"]

# outNames: not the same as anchors,
# these are the Tensors which will be
# connected to the loss layers
outNames = ["out"]

#cifar training data loader : at index 0 : image, at index 1 : label.
cifarInIndices = {"image0": 0}

losses = [poponnx_core.L1Loss("out", "l1LossVal", 0.1)]

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
