import torch
from torchvision import transforms, datasets
import sys
sys.path.append("../../pywillow")
from pywillow import NllLoss, L1Loss, EarlyInfo, TensorInfo, DataFlow
from torchwriter import PytorchNetWriter, conv3x3

from optimizers import SGD

nInChans = 3
nOutChans = 10

# process 2 samples at a time, return requested
# tensors every 6 batches (ie every 12 samples)
# no anchors
batchSize = 2
nBatchesPerCycle = 6
dataFeed = DataFlow(nBatchesPerCycle, batchSize, [])

earlyInfo = EarlyInfo()
earlyInfo.addInfo("image0", TensorInfo("FLOAT", [batchSize, nInChans, 32, 32]))
earlyInfo.addInfo("image1", TensorInfo("FLOAT", [batchSize, nInChans, 32, 32]))
earlyInfo.addInfo("label", TensorInfo("INT64", [batchSize]))

### begin PyTorch


class Module0(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.conv1 = conv3x3(nInChans, nOutChans)
        self.conv2 = conv3x3(nOutChans, nOutChans)
        self.conv3 = conv3x3(nOutChans, nOutChans)
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


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = datasets.CIFAR10(
    root='../../data/cifar10/', train=True, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=4, shuffle=False, num_workers=2)

### end PyTorch


class ModelWriter0(PytorchNetWriter):
    def __init__(self):
        PytorchNetWriter.__init__(
            self,
            inNames=["image0", "image1"],
            outNames=["preProbSquared", "probs"],
            losses=[
                NllLoss("probs", "label", "nllLossVal"),
                L1Loss("preProbSquared", "l1LossVal", 0.01)
            ],
            optimizer=SGD(learnRate=0.001),
            # perform tests, using framework as ground truth
            willowTest=True,
            earlyInfo=earlyInfo,
            dataFeed=dataFeed,
            ### begin PyTorch
            module=Module0(),
            trainloader=trainloader,
            trainLoaderIndices={
                "image0": 0,
                "image1": 0,
                "label": 1
            }
            ### end PyTorch
        )
