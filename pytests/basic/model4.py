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

class Model4(torch.nn.Module):
    def __init__(self, nChans):
        super(Model4, self).__init__()
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
        self.nChans = nChans

    def forward(self, inputs):
        image0 = inputs[0]
        x = conv3x3(self.nChans, self.nChans)(image0) 
        x = conv3x3(self.nChans, self.nChans)(x)
        x_early = self.relu(x)
        x = conv3x3(self.nChans, self.nChans)(x) 
        x = conv3x3(self.nChans, self.nChans)(x) 
        x = conv3x3(self.nChans, self.nChans)(x)
        x = conv3x3(self.nChans, self.nChans)(x) 
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
