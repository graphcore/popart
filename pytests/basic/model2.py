import torch
import sys
sys.path.append("../../pyneuralnet")
from torchwriter import PytorchNetWriter, conv3x3
from losses import NLL, L1
from optimizers import SGD
from datafeeds import FromTxtFiles

class Module2(torch.nn.Module):
    def __init__(self,nInChans, nOutChans):
        torch.nn.Module.__init__(self)
        self.conv1 = conv3x3(nInChans, nOutChans)

    def forward(self, inputs):
        image0 = inputs[0]
        image1 = inputs[1]
        x0 = self.conv1(image0)
        x1 = self.conv1(image1)
        return x0, x1

class ModelWriter2(PytorchNetWriter):
    def __init__(self, nIn, nOut):
        PytorchNetWriter.__init__(
            self,
            inNames=["image0", "image1"],
            outNames=["x0", "x1"],
            losses=[torchwriter.L1(0.1, "x1")],
            optimizer=SGD(learnRate=0.001),
            # as this model has no weights, if we don't include
            # an anchor it will all just be pruned away!
            anchors=["d__image1"],
            # as there are no parameters to train, it must be
            # false otherwise pytorch optimizer freaks out
            neuralnetTest=True,
            dataFeed=FromTxtFiles(
                nSamples=4,
                batchsize=2,
                makeDummy=True,
                streams={
                    "image0": {
                        "file": "./data4mod2/images0.txt",
                        "type": "FLOAT",
                        "shape": [nIn, 32, 32]
                    },
                    "image1": {
                        "file": "./data4mod2/images1.txt",
                        "type": "FLOAT",
                        "shape": [nIn, 32, 32]
                    }
                }),
            module=Module2(nIn, nOut))


