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

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        # interestingly if this is self.conv2,
        # the weights are the same (shared)
        x = self.conv3(x)
        x = self.relu(x)
        window_size = (int(x.size()[2]), int(x.size()[3]))
        print(x.size())
        x = torch.nn.functional.avg_pool2d(x, kernel_size = window_size)
        print(x.size())
        # Not including logsoftmax as it is in the loss we will add, so
        # not using x = self.logsoftmax(x)
        print(x.size())
        print("\n")


        #kernel_size=x.size()[2:], return_indices = False)
        return x


outputdir = sys.argv[1]
if not os.path.exists(outputdir):
    print("Making %s" % (outputdir, ))
    os.mkdir(outputdir)

driver = pydriver.Driver(outputdir)
driver.write(Basic(20, 10), [Variable(torch.rand(2, 20, 32, 32))])
driver.run()

print("pydriver python script complete.")
