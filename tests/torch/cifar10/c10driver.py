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
import tempfile


class TestFailureError(Exception):
    def __init__(self, message):
        super().__init__(message)


def run(torchWriter,
        passes,
        outputdir,
        cifarInIndices,
        device,
        device_hw_id,
        mode="train",
        syntheticData=False,
        transformations=[]):

    poponnx.getLogger().setLevel("TRACE")
    poponnx.getLogger("session").setLevel("WARN")

    if outputdir is None:
        with TemporaryDirectory() as outputdir:
            _run_impl(torchWriter, passes, outputdir, cifarInIndices, device,
                      device_hw_id, mode, syntheticData, transformations)
    else:
        if not os.path.exists(outputdir):
            os.mkdir(outputdir)

        _run_impl(torchWriter, passes, outputdir, cifarInIndices, device,
                  device_hw_id, mode, syntheticData, transformations)


def _run_impl(torchWriter, passes, outputdir, cifarInIndices, device,
              device_hw_id, mode, syntheticData, transformations):
    dataFeed = torchWriter.dataFeed
    inputShapeInfo = torchWriter.inputShapeInfo
    validModes = ["infer", "evaluate", "train"]
    if mode not in validModes:
        raise Exception("mode must be one of " + str(validModes))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # determine what the data directory is
    datadir = "unset"

    dir_path = os.path.dirname(os.path.realpath(__file__))
    path_c10datadir = os.path.join(dir_path, "c10datadir.py")
    if os.path.exists(path_c10datadir):
        import c10datadir
        datadir = c10datadir.c10datadir
    else:
        tmpdir = tempfile.gettempdir()
        datadir = os.path.abspath(os.path.join(tmpdir, 'cifar10data'))
    print("Using datadir=%s" % (datadir))

    if (not os.path.exists(datadir)):
        print(
            "Specified datadir %s does not exist. Consider making it here with os.mkdir(datadir)"
            % (datadir, ))

    print("c10driver: getting data from", datadir)
    trainset = datasets.CIFAR10(
        root=datadir, train=True, download=True, transform=transform)

    fnModel0 = os.path.join(outputdir, "model0.onnx")

    # write ONNX Model to file
    torchWriter.saveModel(fnModel=fnModel0)

    stepLoader = poponnx.DataLoader(
        trainset,
        # the amount of data loaded for each step.
        # note this is not the batch size, it's the "step" size
        # (samples per step)
        batch_size=torchWriter.samplesPerBatch * dataFeed.batchesPerStep(),
        tensor_type='float32',
        #non-random data loading
        shuffle=False,
        num_workers=2)

    deviceManager = poponnx.DeviceManager()

    # Create a CPU device
    if device == "cpu":
        device = deviceManager.createCpuDevice()

    # Create an IPU Model device
    elif device == "ipu_model":

        options = {"compileIPUCode": True, 'numIPUs': 2, "tilesPerIPU": 1216}
        device = deviceManager.createIpuModelDevice(options)

    # Create an Simulator
    elif device == "sim":
        options = {"numIpus": 1, "tilesPerIpu": 1216}
        device = deviceManager.createSimDevice(options)

    # Get a Hardware Device
    elif device == "hw":
        # Get a hardware device that meets the reqirements,
        # may throw if none are available.
        # Will attach to the device
        if device_hw_id:
            device = deviceManager.acquireDeviceById(device_hw_id)
        else:
            device = deviceManager.acquireAvailableDevice()

    # Enumerate available devices
    print("Enumerating devices")
    print("-------------------------------------")
    for idx, d in enumerate(deviceManager.enumerateDevices()):
        print('{0}. {1}'.format(idx, d))
    print("")

    opts = poponnx.SessionOptionsCore()
    opts.ignoreData = syntheticData
    opts.logDir = outputdir

    modelProtoX = fnModel0
    if transformations:
        gc = poponnx.GraphTransformer(fnModel0)
        for transformation in transformations:
            print("Running %s transformation pass" % (transformation, ))
            if transformation == "removeUnusedInputs":
                gc.removeUnusedInputs()

            elif transformation == "prepareNodesForTraining":
                gc.prepareNodesForTraining()

            else:
                raise RuntimeError(
                    "Unrecognised transformation %s" % (transformation, ))

        modelProtoX = gc.getModelProto()

    # Reads ONNX model from file and creates backwards graph,
    # performs Ir optimisations

    if mode == 'infer':
        session = poponnx.InferenceSession(
            fnModel=modelProtoX,
            inputShapeInfo=inputShapeInfo,
            dataFeed=dataFeed,
            passes=passes,
            userOptions=opts,
            deviceInfo=device)
    elif mode == 'evaluate':
        session = poponnx.InferenceSession(
            fnModel=modelProtoX,
            inputShapeInfo=inputShapeInfo,
            dataFeed=dataFeed,
            losses=torchWriter.losses,
            passes=passes,
            userOptions=opts,
            deviceInfo=device)
    else:
        session = poponnx.TrainingSession(
            fnModel=modelProtoX,
            inputShapeInfo=inputShapeInfo,
            dataFeed=dataFeed,
            losses=torchWriter.losses,
            optimizer=torchWriter.optimizer,
            passes=passes,
            userOptions=opts,
            deviceInfo=device)

    # get the tensor info for the anchors
    anchorArrays = session.initAnchorArrays()

    allDotPrefixes = [x[0:-4] for x in os.listdir(outputdir) if ".dot" in x]
    print("Will generate graph pdfs for all of:")
    print(allDotPrefixes)
    import subprocess
    # set generateFromDots to True to
    # generate pdf figures of the Ir. It
    # requires the 'dot' program
    generateFromDots = False
    if generateFromDots:
        for name in allDotPrefixes:
            dotfile = os.path.join(outputdir, "%s.dot" % (name, ))
            outputfile = os.path.join(outputdir, "%s.pdf" % (name, ))
            log = subprocess.call(
                ["dot", "-T", "pdf", "-o", outputfile, dotfile])
            print("Exit status on `%s' was: %s" % (name, log))

    print("Setting device to IPU, and preparing it")
    session.prepareDevice()

    if mode == "train":
        print("Writing weights to device")
        session.weightsFromHost()

        print("Writing Optimizer tensors to device, if there are any")
        session.optimizerFromHost()

    def addStepDimension(data, batchesPerStep):
        if batchesPerStep == 1:
            return data
        else:
            dataShape = np.array(np.shape(data))
            dataShape[0] //= batchesPerStep
            dataShape = np.insert(dataShape, 0, batchesPerStep)
            return np.reshape(data, dataShape)

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
        assertStr = "Tensor" + tId + " must be specified as an anchor"
        assert (tId in anchorArrays.keys()), assertStr
        return anchorArrays[tId]

    def getLossesFromAnchors(torchWriter, anchorArrays):
        # Check all losses are anchored
        for loss in torchWriter.losses:
            lossId = loss.output(0)
            assertStr = "All loss tensors mist be anchored"
            assert (torchWriter.dataFeed.isAnchored(lossId)), assertStr

        # Check all losses have the same anchor return type
        fisrtLossId = torchWriter.losses[0].output(0)
        firstLossArtId = torchWriter.dataFeed.art(fisrtLossId).id()
        if (len(torchWriter.losses) > 1):
            for loss in torchWriter.losses:
                lossId = loss.output(0)
                lossArtId = torchWriter.dataFeed.art(lossId).id()
                assertStr = "All losses must have the same return type"
                assert (lossArtId == firstLossArtId), assertStr

        # Return sum over losses for each sample
        lossShape = np.shape(getAnchorTensor(fisrtLossId, anchorArrays))
        pLosses = np.zeros(lossShape)
        for loss in torchWriter.losses:
            pLosses_ = getAnchorTensor(loss.output(0), anchorArrays)
            pLosses = np.add(pLosses, pLosses_)

        return pLosses

    def subsampleBatches(array, refShape):
        arrayShape = np.shape(array)

        # Every Nth batch
        if len(arrayShape) == len(refShape):
            n = arrayShape[0] // refShape[0]
            return array[n - 1::n]

        # Last batch only
        else:
            return array[-1]

    def getTensorError(tA, pA):
        # pA, tA are corresponding tensors from two models
        pA_shape = np.shape(pA)
        tA_shape = np.shape(tA)
        assert (pA_shape == tA_shape), "Arrays must be same shape"

        ss_err = np.sum((np.array(pA) - np.array(tA))**2)
        ss_pA = np.sum(np.array(pA)**2)
        ss_tA = np.sum(np.array(tA)**2)
        return ss_err / (math.sqrt(ss_pA * ss_tA) + 1.0e-8)

    def checkResult(result, margin):
        if np.isnan(result):
            raise TestFailureError(str(result) + " is NaN")
        elif (result > margin):
            raise TestFailureError(
                str(result) + " is greater than " + str(margin))

    margin = 1.5e-8
    numReports = []

    for epoch in range(4):  # loop over the dataset multiple times
        for stepi, stepData in enumerate(stepLoader):
            if stepi == 1:  # Perform N steps, N=1
                break

            # Form the input map for one step's worth of data.
            # Note: data from the torch DataLoader has shape:
            #   [stepSize * batchSize, sampleShape]
            # whereas Poponnx expects input data of the shape:
            #   [stepSize, batchSize, sampleShape]
            # so we reshape the input array before passing to the stepio
            inputs = {}
            for tenId in cifarInIndices.keys():
                inputs[tenId] = \
                    addStepDimension(stepData[cifarInIndices[tenId]],
                                     session.dataFeed.batchesPerStep())

            if mode == "train":
                # take batchesPerStep passes (1 step), Torch
                torchWriter.train(inputs)

                # take batchesPerStep passes (1 step), PopOnnx
                pystepio = poponnx.PyStepIO(inputs, anchorArrays)
                session.run(pystepio)

                # write models to file
                fnTorchModel = getFnTorch(stepi)
                torchWriter.saveModel(fnTorchModel)
                fnPopOnnxModel = getFnPopOnnx(stepi)
                session.modelToHost(fnPopOnnxModel)

                # Compare parameters from updated Onnx models
                if stepi == 0:
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
                session.run(pystepio)
                pLosses = getLossesFromAnchors(torchWriter, anchorArrays)

                # Compare torch loss tensors with poponnx loss from
                # anchor tensor map.
                # Torch losses returned for all samples, whereas
                # anchors are returned as specified by the user.
                # Subsample torch outputs to match dimensions
                torchLosses = subsampleBatches(torchLosses, np.shape(pLosses))
                result = getTensorError(torchLosses, pLosses)
                print(reportTensorError(0, result))
                checkResult(result, margin)

            elif mode == "infer":
                # take batchesPerStep passes (1 step), Torch
                # returns map of outputs for each sample
                # Note: already are of dimension matching the
                # anchors
                torchOutputs = torchWriter.infer(inputs)

                # take batchesPerStep passes (1 step), PopOnnx
                pystepio = poponnx.PyStepIO(inputs, anchorArrays)
                session.run(pystepio)

                # Compare torch outputs tensors with poponnx output from
                # anchor tensor maps
                for nInd, outName in enumerate(torchWriter.outNames):
                    # Torch outputs returned for all samples, whereas
                    # anchors are returned as specified by the user.
                    # Subsample torch outputs to match dimensions
                    torchOuput = subsampleBatches(
                        torchOutputs[outName], np.shape(anchorArrays[outName]))
                    result = getTensorError(torchOuput, anchorArrays[outName])
                    print(reportTensorError(nInd, result))
                    checkResult(result, margin)
