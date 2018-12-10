import numpy as np
import torch
import torch.utils
import torch.utils.data
import poponnx
from poponnx.writer import NetWriter
from poponnx import TensorInfo, DataFlow, NllLoss, L1Loss, SGD, ConstSGD


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return torch.nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


class PytorchNetWriter(NetWriter):
    def __init__(self, inNames, outNames, losses, optimizer, dataFeed,
                 inputShapeInfo, module):
        """
        module:
          -- pytorch module (whose forward does not have the loss layers)
        all others:
          -- parameters passed to base class.
        """
        NetWriter.__init__(
            self,
            inNames=inNames,
            outNames=outNames,
            losses=losses,
            optimizer=optimizer,
            inputShapeInfo=inputShapeInfo,
            dataFeed=dataFeed)

        self.module = module

    def getTorchOptimizer(self):
        """
        convert poponnx's Optimizer to a torch Optimizer
        """
        if (isinstance(self.optimizer, SGD)
                or isinstance(self.optimizer, ConstSGD)):
            return torch.optim.SGD(
                self.module.parameters(),
                lr=self.optimizer.learnRate(),
                momentum=0.0)
        else:
            raise RuntimeError("unrecognised optimizer")

    def getTorchLossTarget(self, streamMap, outMap):
        """
        Build the torch extension for computing the
        loss described by poponnx's losses
        """
        lossValues = []
        for loss in self.losses:
            if isinstance(loss, NllLoss):
                # note that pytorch documentation has this to
                # say about softmax:
                # Use LogSoftmax instead (it’s faster and has
                # better numerical properties)
                criterion = torch.nn.NLLLoss(reduction="sum")
                logsoftmax = torch.log(outMap[loss.probsTensorId()])
                longlabels = torch.LongTensor(streamMap[loss.labelTensorId()])
                nll_loss = criterion(logsoftmax, longlabels)
                lossValues.append(nll_loss)

            elif isinstance(loss, L1Loss):
                lossValues.append(loss.getLambda() * torch.norm(
                    outMap[loss.getInputId()], 1))

            else:
                raise RuntimeError(
                    "unrecognised loss, cannot get equivalent Torch version")

        return sum(lossValues)

    def saveModel(self, fnModel):
        print("Writing ONNX model to protobuf file %s" % (fnModel, ))
        # jump into eval mode, just to write the onnx model.
        # note that this might do strange things with batch-normalisation (?)
        self.module.eval()

        inputDataInfos = [self.inputShapeInfo.get(tid) for tid in self.inNames]
        inputData = [
            torch.Tensor(np.ones(shape=x.shape(), dtype=x.data_type_lcase()))
            for x in inputDataInfos
        ]

        torch.onnx.export(
            self.module,
            inputData,
            fnModel,
            verbose=False,
            input_names=self.inNames,
            output_names=self.outNames)

    def train(self, inMap):
        """
        Does training
        """
        torchOptimizer = self.getTorchOptimizer()
        self.module.train()
        batchSize = self.dataFeed.batchSize()

        # perform forwards - backwards - update
        # for each of the substeps (substep = batch)

        substepParameterMaps = []
        for substep in range(self.dataFeed.batchesPerStep()):

            substepParameterMap = {}
            substepOutMap = {}
            substepInMap = {}
            for inId in inMap.keys():
                substepInMap[inId] = inMap[inId][substep * batchSize:
                                                 (substep + 1) * batchSize]

            torchOptimizer.zero_grad()
            substepInputs = [
                torch.Tensor(substepInMap[name]) for name in self.inNames
            ]

            # forward pass
            substepOutputs = self.module(substepInputs)

            if len(self.outNames) == 1:
                substepOutMap[self.outNames[0]] = substepOutputs
            else:
                for j, outName in enumerate(self.outNames):
                    substepOutMap[outName] = substepOutputs[j]

            # backwards pass
            lossTarget = self.getTorchLossTarget(substepInMap, substepOutMap)

            lossTarget.backward()
            torchOptimizer.step()

            for name, param in self.module.named_parameters():
                substepParameterMap[name] = param.data
            substepParameterMaps.append(substepParameterMap)

        # returning: list with one entry per substep, of the
        # parameters of the model processed at the substep
        return substepParameterMaps

    def evaluate(self, inMap):
        """
        Does evaluation (i.e. forward pass and loss calculation, but no
        backward pass)
        """
        self.module.eval()

        # perform forwards pass for each of the
        # substeps (substep = batch)

        # in this list we store loss tensors for every Nth
        # sample in the step (as determined by the return
        # period of the anchor tensor)
        losses = []
        stepSize = self.dataFeed.batchesPerStep() * self.dataFeed.batchSize()
        for substep in range(stepSize):

            substepOutMap = {}
            substepInMap = {}
            for inId in inMap.keys():
                substepInMap[inId] = inMap[inId][substep:substep + 1]

            substepInputs = [
                torch.Tensor(substepInMap[name]) for name in self.inNames
            ]

            # forward pass
            substepOutputs = self.module(substepInputs)

            for j, outName in enumerate(self.outNames):
                substepOutMap[outName] = substepOutputs[j]

            # calculate loss, as in backwards pass, but don't update
            # model
            lossTarget = self.getTorchLossTarget(substepInMap, substepOutMap)
            losses.append(lossTarget.item())

        # returning: list with one entry per sample, of the losses
        return losses

    def infer(self, inMap):
        """
        Does inference (i.e. forward pass only)
        """
        self.module.eval()

        # perform forwards pass for each of the
        # substeps (substep = batch)

        # in this map we store a list of the output tensors
        # for every Nth sample in the step (as determined by
        # the return period of the anchor tensor)
        substepOutMap = {}
        for outName in self.outNames:
            substepOutMap[outName] = []
        stepSize = self.dataFeed.batchesPerStep() * self.dataFeed.batchSize()
        for substep in range(stepSize):

            substepInMap = {}
            for inId in inMap.keys():
                substepInMap[inId] = inMap[inId][substep:substep + 1]

            substepInputs = [
                torch.Tensor(substepInMap[name]) for name in self.inNames
            ]

            # forward pass
            substepOutputs = self.module(substepInputs)
            substepOutputTensors = substepOutputs.detach()

            for j, outName in enumerate(self.outNames):
                npTensor = substepOutputTensors[j].numpy()
                substepOutMap[outName].append(npTensor)

        # returning: list with one entry per sample, of the
        # output of the batch
        return substepOutMap
