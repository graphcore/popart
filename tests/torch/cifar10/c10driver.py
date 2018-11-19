import os
import sys
import tempfile
import poponnx
import torch
import numpy as np
import re
from torchvision import transforms, datasets
from poponnx.torch import torchwriter


def run(torchWriter, passes, outputdir, cifarInIndices):

    dataFeed = torchWriter.dataFeed
    earlyInfo = torchWriter.earlyInfo

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    tmpdir = tempfile.gettempdir()
    c10datadir = os.path.abspath(os.path.join(tmpdir, 'cifar10data'))
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
        # the amount of data loaded for each step.
        # note this is not the batch size, it's the "step" size
        # (samples per step)
        batch_size=dataFeed.samplesPerStep(),
        shuffle=False,
        num_workers=3)

    opts = poponnx.SessionOptionsCore()
    opts.exportDot = True
    opts.logging = {"all": "TRACE", "session": "WARN"}

    # Reads ONNX model from file and creates backwards graph,
    # performs Ir optimisations
    session = poponnx.Session(
        fnModel=fnModel0,
        earlyInfo=earlyInfo,
        dataFeed=dataFeed,
        losses=torchWriter.losses,
        optimizer=torchWriter.optimizer,
        outputdir=outputdir,
        passes=passes,
        userOptions=opts)

    # get the tensor info for the anchors
    anchorArrays = session.initAnchorArrays()

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
    session.setDevice("IPU")
    session.prepareDevice()

    print("Writing weights to device")
    session.weightsFromHost()

    print("Writing Optimizer tensors to device, if there are any")
    session.optimizerFromHost()

    def getFnModel(framework, stepi):
        return os.path.join(outputdir, "%sModel_%d.onnx" % (framework, stepi))

    def getFnPopOnnx(stepi):
        return getFnModel("PopOnnx", stepi)

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

            # take batchesPerStep fwd-bwd passes (1 step), PopOnnx
            pystepio = poponnx.PyStepIO(inputs, anchorArrays)
            session.train(pystepio)

            # write models to file, gather comparison statistics
            fnTorchModel = getFnTorch(stepi)
            torchWriter.saveModel(fnTorchModel)
            fnPopOnnxModel = getFnPopOnnx(stepi)
            session.modelToHost(fnPopOnnxModel)

            if stepi == 1:
                numReports.append(
                    poponnx.NumericsReport(fnModel0, fnTorchModel, fnModel0,
                                           fnPopOnnxModel))

            else:
                numReports.append(
                    poponnx.NumericsReport(
                        getFnTorch(stepi - 1), fnTorchModel,
                        getFnPopOnnx(stepi - 1), fnPopOnnxModel))

    for report in numReports:
        print(report.fullReport())

    margin = 1.0e-8

    for report in numReports:
        freport = report.fullReport()
        matches = re.findall('= (\S*)', freport)
        for match in matches:
            result = float(match)
            print("Result = " + str(result))

            # Raise an error if the result is greater than the margin
            if (result > margin):
                raise TestFailureError(
                    str(result) + " is greater than " + str(margin))
