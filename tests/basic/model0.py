import torch
import sys
sys.path.append("../../pywillow")
import pywillow
from torchwriter import PytorchNetWriter, conv3x3
from optimizers import SGD
from datafeeds import FromTxtFiles


class Module0(torch.nn.Module):
    def __init__(self, inChans, outChans):
        torch.nn.Module.__init__(self)
        self.conv1 = conv3x3(inChans, outChans)
        self.conv2 = conv3x3(outChans, outChans)
        self.conv3 = conv3x3(outChans, outChans)
        self.relu = torch.nn.functional.relu
        self.logsoftmax = torch.nn.LogSoftmax(dim=0)

    def forward(self, inputs):
        image0 = inputs[0]
        image1 = inputs[1]
        x = image0 + image1

        x = self.conv1(x)
        x = self.relu(x)
        # interestingly if this is self.conv2,
        # the weights are the same (shared)
        x = self.conv3(x)
        preProbSquared = x + x

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


class ModelWriter0(PytorchNetWriter):
    def __init__(self, inChans, outChans):
        PytorchNetWriter.__init__(
            self,
            inNames=["image0", "image1"],
            outNames=["preProbSquared", "probs"],
            losses=[pywillow.NllLoss("probs", "label", "nllLossVal"),
                    pywillow.L1Loss("preProbSquared", "l1LossVal", 0.01)],
            optimizer=SGD(learnRate=0.001),
            anchors=[],
            willowTest=True,
            dataFeed=FromTxtFiles(
                nSamples=12,
                batchsize=2,
                makeDummy=True,
                streams={
                    "image0": {
                        "file": "../../data/data/images0.txt",
                        "type": "FLOAT",
                        "shape": [inChans, 8, 8]
                    },
                    "image1": {
                        "file": "../../data/data/images1.txt",
                        "type": "FLOAT",
                        "shape": [inChans, 8, 8]
                    },
                    # a label is a scalar, hence the []
                    "label": {
                        "file": "../../data/data/label.txt",
                        "type": "INT64",
                        "shape": []
                    },
                }),
            # and finally the pytorch specific part, 
            # (everything til now is generic NetWriter)
            module=Module0(inChans, outChans))
