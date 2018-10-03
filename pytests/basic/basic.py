import os
import sys
from torch.autograd import Variable
import torch.nn
import torch.nn.functional
import torchvision
sys.path.append("../../driver")
import pydriver
import importlib
importlib.reload(pydriver)
import subprocess

if (len(sys.argv) != 2):
    raise RuntimeError("onnx_net.py <log directory>")

# define pytorch model


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return torch.nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


model_number = 4


class Basic0(torch.nn.Module):
    def __init__(self, inChans, outChans):
        super(Basic0, self).__init__()
        self.conv1 = conv3x3(inChans, outChans)
        self.conv2 = conv3x3(outChans, outChans)
        self.relu = torch.nn.functional.relu
        self.conv3 = conv3x3(outChans, outChans)
        self.logsoftmax = torch.nn.LogSoftmax(dim=0)
        self.softmax = torch.nn.Softmax(dim=0)
        # specific to project neuralnet
        self.output_names = ["preProbSquared", "probs"]
        self.losses = [
            pydriver.NLL("probs", "labels"),
            pydriver.L1(0.1, "preProbSquared")
        ]
        self.input_names = ["image0", "image1"]
        self.anchors = []
        self.inputs = [
            torch.rand(2, nInChans, 32, 32),
            torch.rand(2, nInChans, 32, 32)
        ]

    def forward(self, inputs):
        image0 = inputs[0]
        image1 = inputs[1]
        x2 = self.relu(image1)
        x = image0 + x2
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        # interestingly if this is self.conv2,
        # the weights are the same (shared)
        x = self.conv3(x)
        preProbSquared = x + x
        x = self.relu(x)
        window_size = (int(x.size()[2]), int(x.size()[3]))
        x = torch.nn.functional.avg_pool2d(x, kernel_size=window_size)
        x = torch.squeeze(x)
        # probabilities:
        probs = self.logsoftmax(x)
        # -> currently no support from pytorch
        # -> for gather or log (pytorch 0.4.1)
        # x = torch.gather(input = x, dim = 1, index= labels)
        # loss = torch.log(x)
        return preProbSquared, probs


class Basic3(torch.nn.Module):
    def __init__(self, nChans):
        super(Basic3, self).__init__()
        self.logsoftmax = torch.nn.LogSoftmax(dim=0)
        # specific to project neuralnet
        self.output_names = ["probs"]
        self.losses = [
            pydriver.NLL("probs", "labels")
        ]
        self.input_names = ["image0"]
        self.anchors = []
        self.inputs = [
            torch.rand(2, nChans, 32, 32),
        ]

    def forward(self, inputs):
        image0 = inputs[0]
        x = conv3x3(nChans, nChans)(image0) 
        x = conv3x3(nChans, nChans)(x)
        x = conv3x3(nChans, nChans)(x) 
        x = conv3x3(nChans, nChans)(x) 
        x = conv3x3(nChans, nChans)(x)
        x = conv3x3(nChans, nChans)(x) 
        window_size = (int(x.size()[2]), int(x.size()[3]))
        x = torch.nn.functional.avg_pool2d(x, kernel_size=window_size)
        x = torch.squeeze(x)
        # probabilities:
        probs = self.logsoftmax(x)
        return probs


class Basic4(torch.nn.Module):
    def __init__(self, nChans):
        super(Basic4, self).__init__()
        # specific to project neuralnet
        self.output_names = ["Y"]
        self.losses = [
            pydriver.L1(0.1,"Y")
        ]
        self.input_names = ["image0"]
        self.anchors = []
        self.inputs = [
            torch.rand(2, nChans, 25, 4),
        ]
        self.relu = torch.nn.functional.relu

    def forward(self, inputs):
        image0 = inputs[0]
        x = conv3x3(nChans, nChans)(image0) 
        x = conv3x3(nChans, nChans)(x)
        x_early = self.relu(x)
        x = conv3x3(nChans, nChans)(x) 
        x = conv3x3(nChans, nChans)(x) 
        x = conv3x3(nChans, nChans)(x)
        x = conv3x3(nChans, nChans)(x) 
        x0 = self.relu(x)
        x1 = self.relu(x)
        x2 = self.relu(x)
        x3 = self.relu(x)
        x01 = x0 + x1
        x23 = x2 + x3
        x0123 = x01 + x23
        x = self.relu(x0123) + x_early
        y = x + x
        return y



class Basic1(torch.nn.Module):
    def __init__(self):
        super(Basic1, self).__init__()
        self.output_names = ["x0", "x1"]
        self.losses = [pydriver.L1(0.1, "x1")]
        self.input_names = ["image0", "image1"]
        self.anchors = []
        self.inputs = [torch.rand(2, 3, 4, 5), torch.rand(2, 3, 4, 5)]

    def forward(self, inputs):
        image0 = inputs[0]
        image1 = inputs[1]
        x0 = image0 + image0
        x1 = image0 + image1
        return x0, x1


class Basic2(torch.nn.Module):
    def __init__(self, inChans, outChans):
        super(Basic2, self).__init__()
        self.conv1 = conv3x3(inChans, outChans)
        self.output_names = ["x0", "x1"]
        self.losses = [pydriver.L1(0.1, "x1")]
        self.input_names = ["image0", "image1"]
        self.anchors = ["d__image1"]
        self.inputs = [
            torch.rand(2, nInChans, 32, 32),
            torch.rand(2, nInChans, 32, 32)
        ]

    def forward(self, inputs):
        image0 = inputs[0]
        image1 = inputs[1]
        x0 = self.conv1(image0)
        x1 = self.conv1(image1)
        return x0, x1


outputdir = sys.argv[1]
if not os.path.exists(outputdir):
    print("Making %s" % (outputdir, ))
    os.mkdir(outputdir)

model = None
if model_number == 0:
    nInChans = 20
    nOutChans = 10
    model = Basic0(nInChans, nOutChans)
elif model_number == 1:
    model = Basic1()
elif model_number == 2:
    nInChans = 20
    nOutChans = 10
    model = Basic2(nInChans, nOutChans)
elif model_number == 3:
    nChans = 25
    model = Basic3(nChans)
elif model_number == 4:
    nChans = 5
    model = Basic4(nChans)

else:
    raise RuntimeError("invalid model number")

driver = pydriver.Driver(outputdir)
driver.write(
    model,
    inputs=model.inputs,
    input_names=model.input_names,
    output_names=model.output_names,
    anchors=model.anchors,
    losses=model.losses, 
    outputdir=outputdir)
driver.run()

dotfile = os.path.join(outputdir, "jam.dot")
outputfile = os.path.join(outputdir, "jam.pdf")
print("generating %s"%(outputfile,))
#dotgenline = "dot -T -o %s %s"%(outputfile, dotfile,)
log = subprocess.call(["dot", "-T", "pdf", "-o", outputfile, dotfile])
print(log)

print("pydriver python script complete.")
