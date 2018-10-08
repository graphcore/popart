import torch
import sys
sys.path.append("../../pywillow")
from torchwriter import PytorchNetWriter, conv3x3
from losses import NLL, L1
from optimizers import SGD
from datafeeds import FromTxtFiles


class Module4(torch.nn.Module):
    def __init__(self, nChans):
        super(Module4, self).__init__()
        self.relu = torch.nn.functional.relu
        self.nChans = nChans
        self.conv1 = conv3x3(self.nChans, self.nChans)
        self.conv2 = conv3x3(self.nChans, self.nChans)
        self.conv3 = conv3x3(self.nChans, self.nChans)
        self.conv4 = conv3x3(self.nChans, self.nChans)
        self.conv5 = conv3x3(self.nChans, self.nChans)
        self.conv6 = conv3x3(self.nChans, self.nChans)

    def forward(self, inputs):
        image0 = inputs[0]
        x = self.conv1(image0)
        x = self.conv2(x)
        x_early = self.relu(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
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


class ModelWriter4(PytorchNetWriter):
    def __init__(self, nChans):
        PytorchNetWriter.__init__(
            self,
            inNames=["image0"],
            outNames=["Y"],
            losses=[L1(0.1, "Y")],
            optimizer=SGD(learnRate=0.001),
            anchors=[],
            dataFeed=FromTxtFiles(
                nSamples=12,
                batchsize=2,
                makeDummy=True,
                streams={
                    "image0": {
                        "file": "./data/images0.txt",
                        "type": "FLOAT",
                        "shape": [nChans, 25, 4]
                    },
                }),
            # and finally the pytorch specific part:
            module=Module4(nChans))
