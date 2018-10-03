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

class Model2(torch.nn.Module):
    def __init__(self, inChans, outChans):
        super(Model2, self).__init__()
        self.conv1 = conv3x3(inChans, outChans)
        self.output_names = ["x0", "x1"]
        self.losses = [pydriver.L1(0.1, "x1")]
        self.input_names = ["image0", "image1"]
        self.anchors = ["d__image1"]
        self.inputs = [
            torch.rand(2, inChans, 32, 32),
            torch.rand(2, inChans, 32, 32)
        ]

    def forward(self, inputs):
        image0 = inputs[0]
        image1 = inputs[1]
        x0 = self.conv1(image0)
        x1 = self.conv1(image1)
        return x0, x1
