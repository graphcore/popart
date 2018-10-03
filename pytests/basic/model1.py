import torch
import sys
sys.path.append("../../driver")
import pydriver
import importlib
importlib.reload(pydriver)

class Model1(torch.nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        self.output_names = ["x0", "x1"]
        self.losses = [pydriver.L1(0.1, "x1")]
        self.input_names = ["image0", "image1"]
        # as this model has no weights, if we don't include 
        # an anchor it will all just be pruned away!
        self.anchors = ["d__image0"]
        self.inputs = [torch.rand(2, 3, 4, 5), torch.rand(2, 3, 4, 5)]

    def forward(self, inputs):
        image0 = inputs[0]
        image1 = inputs[1]
        x0 = image0 + image0
        x1 = image0 + image1
        return x0, x1

