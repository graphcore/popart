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

class Model3(torch.nn.Module):
    def __init__(self, nChans):
        super(Model3, self).__init__()
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
        self.nChans = nChans


    def forward(self, inputs):
        image0 = inputs[0]
        x = conv3x3(self.nChans, self.nChans)(image0)
        x = conv3x3(self.nChans, self.nChans)(x)
        x = conv3x3(self.nChans, self.nChans)(x)
        x = conv3x3(self.nChans, self.nChans)(x)
        x = conv3x3(self.nChans, self.nChans)(x)
        x = conv3x3(self.nChans, self.nChans)(x)
        window_size = (int(x.size()[2]), int(x.size()[3]))
        x = torch.nn.functional.avg_pool2d(x, kernel_size=window_size)
        x = torch.squeeze(x)
        # probabilities:
        probs = self.logsoftmax(x)
        return probs
