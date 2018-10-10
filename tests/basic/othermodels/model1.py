import torch
import sys
sys.path.append("../../pywillow")
from pywillow import L1Loss, SGD
from torchwriter import PytorchNetWriter, conv3x3


class Module1(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)

    def forward(self, inputs):
        image0 = inputs[0]
        image1 = inputs[1]
        x0 = image0 + image0
        x1 = image0 + image1
        return x0, x1


class ModelWriter1(PytorchNetWriter):
    def __init__(self):
        PytorchNetWriter.__init__(
            self,
            inNames=["image0", "image1"],
            outNames=["x0", "x1"],
            losses=[L1Loss("x1", "l1LossVal", 0.1)],
            optimizer=SGD(learnRate=0.001),
            # as this model has no weights, if we don't include
            # an anchor it will all just be pruned away!
            anchors=["d__image0"],
            # as there are no parameters to train, it must be
            # false otherwise pytorch optimizer freaks out
            willowTest=False,
            dataFeed=FromTxtFiles(
                nSamples=12,
                batchsize=2,
                makeDummy=True,
                streams={
                    "image0": {
                        "file": "../../data/data4mod1/images0.txt",
                        "type": "FLOAT",
                        "shape": [3, 4, 5]
                    },
                    "image1": {
                        "file": "../../data/data4mod1/images1.txt",
                        "type": "FLOAT",
                        "shape": [3, 4, 5]
                    }
                }),
            module=Module1())
