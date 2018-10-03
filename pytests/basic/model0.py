import torch
import sys
sys.path.append("../../driver")
import pydriver
import importlib
importlib.reload(pydriver)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return torch.nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)

class Model0(torch.nn.Module):
    def __init__(self, inChans, outChans):
        super(Model0, self).__init__()
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
            torch.rand(2, inChans, 32, 32),
            torch.rand(2, inChans, 32, 32)
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
