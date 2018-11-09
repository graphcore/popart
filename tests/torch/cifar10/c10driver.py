import os
import sys

## Search paths for poponnx .so/.dylib and .py files
# TODO this needs to be removed using __init__.py or some similar scheme
testdir = os.path.dirname(os.path.abspath(__file__))
# .so/.dylib file
libpath = os.path.abspath(os.path.join(testdir, "../../../lib"))
sys.path.append(libpath)
# .py files
pypath = os.path.abspath(os.path.join(testdir, "../../../python"))
sys.path.append(pypath)

if sys.platform != "darwin":
    # So python finds poponnx.so when importing poponnx (for Ubuntu)
    # (without having to export LD_LIBRARY_PATH)
    import ctypes
    ctypes.cdll.LoadLibrary(os.path.join(libpath, "libpoponnx.so"))

import torch
import numpy as np
from torchvision import transforms, datasets
import poponnx_core
from poponnx.torch import torchwriter

def run(torchWriter, willowOptPasses, outputdir, cifarInIndices):

    dataFeed = torchWriter.dataFeed
    earlyInfo = torchWriter.earlyInfo

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    c10datadir = os.path.abspath(os.path.join(outputdir, 'cifar10data'))
    if (not os.path.exists(c10datadir)):
        print("Creating directory %s" % (c10datadir))
        os.mkdir(c10datadir)

    trainset = datasets.CIFAR10(
        root=c10datadir, train=True, download=True, transform=transform)

    fnModel0 = os.path.join(outputdir, "model0.onnx")

    # write ONNX Model to file
    torchWriter.saveModel(fnModel=fnModel0)

    stepLoader = torch.utils.data.DataLoader(
        trainset,
        # the amount of data loaded for each willow step.
        # note this is not the batch size, it's the "step" size
        # (samples per step)
        batch_size=dataFeed.samplesPerStep(),
        shuffle=False,
        num_workers=3)

    # Reads ONNX model from file and creates backwards graph,
    # performs Ir optimisations
    willowNet = poponnx_core.Net(
        fnModel0, earlyInfo, dataFeed, torchWriter.losses,
        torchWriter.optimizer, [], outputdir, willowOptPasses)

    # get the tensor info for the anchors
    willowAnchorArrays = {}
    for anchor in dataFeed.anchors():
        x = willowNet.getInfo(anchor)
        outShape = x.shape()
        # Note : == is not the same as "is" here.
        if dataFeed.art() == poponnx_core.AnchorReturnType.ALL:
            outShape[0] = outShape[0] * dataFeed.batchesPerStep()
        elif dataFeed.art() == poponnx_core.AnchorReturnType.SUM:
            outShape[0] = outShape[0] / dataFeed.batchesPerStep()
        elif dataFeed.art() == poponnx_core.AnchorReturnType.FINAL:
            outShape[0] = outShape[0]
        else:
            raise RuntimeError("unrecognised AnchorType")
        willowAnchorArrays[anchor] = 7 * np.ones(
            shape=outShape, dtype=x.data_type_lcase())

    allDotPrefixes = [x[0:-4] for x in os.listdir(outputdir) if ".dot" in x]
    print("Will generate graph pdfs for all of:")
    print(allDotPrefixes)
    import subprocess
    for name in allDotPrefixes:
        dotfile = os.path.join(outputdir, "%s.dot" % (name, ))
        outputfile = os.path.join(outputdir, "%s.pdf" % (name, ))
        #log = subprocess.call(["dot", "-T", "pdf", "-o", outputfile, dotfile])
        #print("Exit status on `%s' was: %s" % (name, log))
    print("torchWriter calling script complete.")

    print("Setting device to IPU, and preparing it")
    willowNet.setDevice("IPU")
    willowNet.prepareDevice()

    print("Writing weights to device")
    willowNet.weightsFromHost()

    print("Writing Optimizer tensors to device, if there are any")
    willowNet.optimizerFromHost()

    def getFnModel(framework, stepi):
        return os.path.join(outputdir, "%sModel_%d.onnx" % (framework, stepi))

    def getFnWillow(stepi):
        return getFnModel("Willow", stepi)

    def getFnTorch(stepi):
        return getFnModel("Torch", stepi)

    stepi = 0
    numReports = []
    for epoch in range(4):  # loop over the dataset multiple times
        for i, data in enumerate(stepLoader, 0):
            if i == 1:
                break

            images, labels = data

            inputs = {}
            for tenId in cifarInIndices.keys():
                inputs[tenId] = data[cifarInIndices[tenId]].numpy()
            stepi += 1

            # take batchesPerStep fwd-bwd passes (1 step), Torch
            torchOutputs = torchWriter.step(inputs)

            # take batchesPerStep fwd-bwd passes (1 step), Willow
            pystepio = poponnx_core.PyStepIO(inputs, willowAnchorArrays)
            willowNet.step(pystepio)

            # write models to file, gather comparison statistics
            fnTorchModel = getFnTorch(stepi)
            torchWriter.saveModel(fnTorchModel)
            fnWillowModel = getFnWillow(stepi)
            willowNet.modelToHost(fnWillowModel)

            if stepi == 1:
                numReports.append(
                    poponnx_core.NumericsReport(fnModel0, fnTorchModel,
                                                fnModel0, fnWillowModel))

            else:
                numReports.append(
                    poponnx_core.NumericsReport(
                        getFnTorch(stepi - 1), fnTorchModel,
                        getFnWillow(stepi - 1), fnWillowModel))

    for report in numReports:
        print(report.fullReport())
