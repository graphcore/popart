import os
import sys
import tempfile
import poponnx
import torch
import numpy as np
import math
import re
from tempfile import TemporaryDirectory
from torchvision import transforms, datasets
from poponnx.torch import torchwriter
from poponnx import NllLoss, L1Loss


class TestFailureError(Exception):
    def __init__(self, message):
        super().__init__(message)


def run(torchWriter, passes, outputdir, cifarInIndices, mode="train"):
    if outputdir is None:
        with TemporaryDirectory() as outputdir:
            _run_impl(torchWriter, passes, outputdir, cifarInIndices, mode)
    else:
        if not os.path.exists(outputdir):
            os.mkdir(outputdir)

        _run_impl(torchWriter, passes, outputdir, cifarInIndices, mode)


def _run_impl(torchWriter, passes, outputdir, cifarInIndices, mode):
    dataFeed = torchWriter.dataFeed
    earlyInfo = torchWriter.earlyInfo
    validModes = ["infer", "evaluate", "train"]
    if mode not in validModes:
        raise Exception("mode must be one of " + str(validModes))

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
        batch_size=dataFeed.batchSize() * dataFeed.batchesPerStep(),
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

    def reportTensorError(tensorInd, result):
        reportStr = str(tensorInd) + " :\n"
        reportStr += "  |pA - tA|^2 / (|pA||tA| + 1e-8)  = " + str(
            result) + "\n"
        return reportStr

    def getAnchorTensor(tId, anchorArrays):
        assertStr = "Loss tensor must be specified as an anchor"
        assert (tId in anchorArrays.keys()), assertStr
        return anchorArrays[loss.output(0)]

    def getTensorError(pA, tA):
        # pA, tA are corresponding tensors from two models
        pA_shape = np.shape(pA)
        tA_shape = np.shape(tA)
        assert (pA_shape == tA_shape), "Arrays must be same shape"

        ss_err = np.sum((np.array(pA) - np.array(tA))**2)
        ss_pA = np.sum(np.array(pA)**2)
        ss_tA = np.sum(np.array(tA)**2)
        return ss_err / (math.sqrt(ss_pA * ss_tA) + 1.0e-8)

    def checkResult(result, margin):
        if (result > margin):
            raise TestFailureError(
                str(result) + " is greater than " + str(margin))

    margin = 1.0e-8
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

            if mode == "train":
                # take batchesPerStep passes (1 step), Torch
                torchWriter.train(inputs)

                # take batchesPerStep passes (1 step), PopOnnx
                pystepio = poponnx.PyStepIO(inputs, anchorArrays)
                session.train(pystepio)

                # write models to file
                fnTorchModel = getFnTorch(stepi)
                torchWriter.saveModel(fnTorchModel)
                fnPopOnnxModel = getFnPopOnnx(stepi)
                session.modelToHost(fnPopOnnxModel)

                # Compare parameters from updated Onnx models
                if stepi == 1:
                    nr = poponnx.NumericsReport(fnModel0, fnTorchModel,
                                                fnModel0, fnPopOnnxModel)
                else:
                    nr = poponnx.NumericsReport(
                        getFnTorch(stepi - 1), fnTorchModel,
                        getFnPopOnnx(stepi - 1), fnPopOnnxModel)

                print(nr.fullReport())
                # One relative error calculated per weight tensor
                for tId, relerror in nr.getRelativeErrors().items():
                    checkResult(relerror, margin)

            elif mode == "evaluate":
                # take batchesPerStep passes (1 step), Torch
                # returns scalar for each sample
                torchLosses = torchWriter.evaluate(inputs)

                # take batchesPerStep passes (1 step), PopOnnx
                pystepio = poponnx.PyStepIO(inputs, anchorArrays)
                session.evaluate(pystepio)

                # Compare torch loss tensors with poponnx loss from
                # anchor tensor map
                pLoss = np.zeros(stepLoader.batch_size)
                for loss in torchWriter.losses:
                    pLoss_ = getAnchorTensor(loss.output(0), anchorArrays)
                    pLoss = np.add(pLoss, pLoss_)
                result = getTensorError(torchLosses, pLoss)
                print(reportTensorError(0, result))
                checkResult(result, margin)

            elif mode == "infer":
                # take batchesPerStep passes (1 step), Torch
                # returns map of outputs for each sample
                torchOutputs = torchWriter.infer(inputs)

                # take batchesPerStep passes (1 step), PopOnnx
                pystepio = poponnx.PyStepIO(inputs, anchorArrays)
                session.infer(pystepio)

                # Compare torch outputs tensors with poponnx output from
                # anchor tensor maps
                for nInd, outName in enumerate(torchWriter.outNames):
                    result = getTensorError(torchOutputs[outName],
                                            anchorArrays[outName])
                    print(reportTensorError(nInd, result))
                    checkResult(result, margin)
