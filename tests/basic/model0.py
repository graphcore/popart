import torch
from torchvision import transforms, datasets
import sys
import os
sys.path.append("../../pywillow")
from pywillow import NllLoss, L1Loss, EarlyInfo, TensorInfo, PyStepIO
from pywillow import DataFlow, SGD, ConstSGD, WillowNet, getTensorInfo
from pywillow import AnchorReturnType
from torchwriter import PytorchNetWriter, conv3x3

if (len(sys.argv) != 2):
    raise RuntimeError("onnx_net.py <log directory>")

outputdir = sys.argv[1]
if not os.path.exists(outputdir):
    print("Making %s" % (outputdir, ))
    os.mkdir(outputdir)

nInChans = 3
nOutChans = 10

# process samplesPerBatch = 2 samples at a time,
# so weights updated on average gradient of samplesPerBatch = 2
# samples: samplesPerBatch is EXACTLY the standard batch size.
samplesPerBatch = 2
# Return requested tensors every batchesPerStep = 3 cycles.
# so (ie only communicate back to host every 2*3 = 6 samples)
batchesPerStep = 3
# anchors : return the losses
anchors = ["nllLossVal", "l1LossVal"]
# what to return. See relevant poplar headers for option descriptions
art = AnchorReturnType.ALL

dataFeed = DataFlow(batchesPerStep, samplesPerBatch, anchors, art)

earlyInfo = EarlyInfo()
earlyInfo.add("image0", TensorInfo("FLOAT",
                                   [samplesPerBatch, nInChans, 32, 32]))
earlyInfo.add("image1", TensorInfo("FLOAT",
                                   [samplesPerBatch, nInChans, 32, 32]))
earlyInfo.add("label", TensorInfo("INT32", [samplesPerBatch]))

inNames = ["image0", "image1"]
outNames = ["preProbSquared", "probs"]
losses = [
    NllLoss("probs", "label", "nllLossVal"),
    L1Loss("preProbSquared", "l1LossVal", 0.01)
]

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

# To take a few training steps in Pytorch, for verification purposes
verifTrainLoader = torch.utils.data.DataLoader(
    trainset, batch_size=samplesPerBatch, shuffle=False, num_workers=2)

### end PyTorch

willowTrainLoader = torch.utils.data.DataLoader(
    trainset,
    # the amount of data loaded for each willow step
    # note this is not the batch size, it's the "step" size
    batch_size=samplesPerBatch * batchesPerStep,
    shuffle=False,
    num_workers=2)


class ModelWriter0(PytorchNetWriter):
    def __init__(self):
        PytorchNetWriter.__init__(
            self,
            inNames=inNames,
            outNames=outNames,
            losses=losses,
            optimizer=SGD(0.001),
            earlyInfo=earlyInfo,
            dataFeed=dataFeed,
            # perform tests, using framework as ground truth
            willowVerif=True,
            ### begin PyTorch
            module=Module0(),
            trainloader=verifTrainLoader,
            trainLoaderIndices={
                "image0": 0,
                "image1": 0,
                "label": 1
            }
            ### end PyTorch
        )


# write to file(s)
writer = ModelWriter0()
writer.write(dirname=outputdir)

# C++ class reads from file(s) and creates backwards graph
pynet = WillowNet(
    os.path.join(outputdir, "model0.onnx"),
    writer.earlyInfo,
    writer.dataFeed,
    writer.losses,
    writer.optimizer,
    [],
    outputdir,
    ["PreUniRepl", "PostNRepl", "LsmGradDirect"
     ]  # The optimization passes to run, see patterns.hpp
)

allDotPrefixes = [x[0:-4] for x in os.listdir(outputdir) if ".dot" in x]
print("Will generate graph pdfs for all of:")
print(allDotPrefixes)
import subprocess
for name in allDotPrefixes:
    dotfile = os.path.join(outputdir, "%s.dot" % (name, ))
    outputfile = os.path.join(outputdir, "%s.pdf" % (name, ))
    log = subprocess.call(["dot", "-T", "pdf", "-o", outputfile, dotfile])
    print("Exit status on `%s' was: %s" % (name, log))
print("torchwriter calling script complete.")

pynet.setDevice("IPU")
pynet.prepareDevice()
pynet.weightsFromHost()
pynet.optimizerFromHost()

for epoch in range(2):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(willowTrainLoader, 0):
        if i == 5:
            break

        images, labels = data
        inputs = {
            "image0": images.numpy(),
            "image1": images.numpy(),
            "label": labels.numpy()
        }
        # TODO : create output tensors
        outputs = {}
        pystepio = PyStepIO(inputs, outputs)
        pynet.step(pystepio)

#saveModel(fileNameForModelWrite)
