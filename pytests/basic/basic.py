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

class Basic(torch.nn.Module):
    def __init__(self, inChans, outChans):
        super(Basic, self).__init__()
        self.conv1 = conv3x3(inChans, outChans)
        self.conv2 = conv3x3(outChans, outChans)
        self.relu = torch.nn.functional.relu
        self.conv3 = conv3x3(outChans, outChans)
        self.logsoftmax = torch.nn.LogSoftmax(dim=0)
        self.softmax = torch.nn.Softmax(dim=0)

    def forward0(self, inputs):
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
        x = torch.nn.functional.avg_pool2d(x, kernel_size = window_size)
        x = torch.squeeze(x)
        # probabilities:
        probs = self.logsoftmax(x)
        # -> currently no support from pytorch
        # -> for gather or log (pytorch 0.4.1)
        # x = torch.gather(input = x, dim = 1, index= labels)
        # loss = torch.log(x)

        return preProbSquared, probs

    def forward1(self, inputs):
        image0 = inputs[0]
        image1 = inputs[1]
        x0 = image0 + image0
        x1 = image0 + image1
        return x0, x1


    def forward(self, inputs):
        image0 = inputs[0]
        image1 = inputs[1]
        x0 = self.conv1(image0)
        x1 = self.conv1(image1)
        return x0, x1




output_names_0 = ["preProbSquared", "probs"]
losses_0 = [pydriver.NLL("probs", "labels"), pydriver.L1(0.1, "preProbSquared")]
output_names_0 = ["x0", "x1"]

output_names_1 = ["x0", "x1"]
losses_1 = [pydriver.L1(0.1, "x1")]
input_names_1 = ["image0", "image1"]


output_names = ["x0", "x1"]
losses = [pydriver.L1(0.1, "x0"), pydriver.L1(0.1, "x1")]
input_names = ["image0", "image1"]


outputdir = sys.argv[1]
if not os.path.exists(outputdir):
    print("Making %s" % (outputdir, ))
    os.mkdir(outputdir)

nInChans = 20
nOutChans = 10
driver = pydriver.Driver(outputdir)
driver.write(
    Basic(nInChans, nOutChans),
    [torch.rand(2, nInChans, 32, 32),
     torch.rand(2, nInChans, 32, 32)],
    input_names=input_names,
    output_names=output_names,
    losses=losses)
driver.run()

print("pydriver python script complete.")
