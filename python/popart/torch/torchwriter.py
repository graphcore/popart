# Copyright (c) 2018 Graphcore Ltd. All rights reserved.
import numpy as np
import torch
import torch.utils
import torch.utils.data
import popart
from popart.writer import NetWriter
from popart import TensorInfo, DataFlow, IdentityLoss, SGD, ConstSGD


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return torch.nn.Conv2d(in_planes,
                           out_planes,
                           kernel_size=3,
                           stride=stride,
                           padding=1,
                           bias=False)


class PytorchNetWriter(NetWriter):
    def __init__(self, inNames, outNames, losses, optimizer, dataFeed,
                 inputShapeInfo, module, samplesPerBatch):
        """
        module:
          -- pytorch module (whose forward does not have the loss layers)
        all others:
          -- parameters passed to base class.
        """
        NetWriter.__init__(self,
                           inNames=inNames,
                           outNames=outNames,
                           losses=losses,
                           optimizer=optimizer,
                           inputShapeInfo=inputShapeInfo,
                           dataFeed=dataFeed)

        self.module = module
        self.samplesPerBatch = samplesPerBatch

    def getTorchOptimizer(self):
        """
        convert popart's Optimizer to a torch Optimizer
        """
        if (isinstance(self.optimizer, SGD)
                or isinstance(self.optimizer, ConstSGD)):
            return torch.optim.SGD(
                self.module.parameters(),
                lr=self.optimizer.learningRates().getDefault().val(),
                weight_decay=self.optimizer.weightDecays().getDefault().val(),
                momentum=self.optimizer.momentums().getDefault().val())
        else:
            raise RuntimeError("unrecognised optimizer")

    def getTorchLossTarget(self, streamMap, outMap):
        """
        Build the torch extension for computing the
        loss described by popart's losses
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
                lossValues.append(loss.getLambda() *
                                  torch.norm(outMap[loss.getInputId()], 1))

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
            torch.from_numpy(
                np.ones(shape=x.shape(), dtype=x.data_type_lcase()))
            for x in inputDataInfos
        ]

        torch.onnx.export(self.module,
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

        # if batchesPerStep is 1, a dimension will be missing
        if self.dataFeed.batchesPerStep() == 1:
            inMap = _add_dimension(inMap)

        # perform forwards - backwards - update
        # for each of the substeps (substep = batch)

        stepParameterMap = []
        for substepi in range(self.dataFeed.batchesPerStep()):

            substepParameterMap = {}
            substepOutMap = {}
            substepInMap = {}
            for inId in inMap.keys():
                substepInMap[inId] = inMap[inId][substepi][0:]

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
            stepParameterMap.append(substepParameterMap)

        # returning: list with one entry per substep, of the
        # parameters of the model processed at the substep
        return stepParameterMap

    def evaluate(self, inMap):
        """
        Does evaluation (i.e. forward pass and loss calculation, but no
        backward pass)
        """
        self.module.eval()

        # if batchesPerStep is 1, a dimension will be missing
        if self.dataFeed.batchesPerStep() == 1:
            inMap = _add_dimension(inMap)

        # perform forwards pass for each batch
        losses = []
        for substepi in range(self.dataFeed.batchesPerStep()):
            sampleLosses = []

            for samplei in range(self.samplesPerBatch):
                sampleInMap = {}
                sampleOutMap = {}

                for inId in inMap.keys():
                    sampleInMap[inId] = inMap[inId][substepi][samplei:samplei +
                                                              1]

                sampleInputs = [
                    torch.Tensor(sampleInMap[name]) for name in self.inNames
                ]

                # forward pass
                sampleOutputs = self.module(sampleInputs)

                if len(self.outNames) == 1:
                    sampleOutMap[self.outNames[0]] = sampleOutputs
                else:
                    for j, outName in enumerate(self.outNames):
                        sampleOutMap[outName] = sampleOutputs[j]

                # calculate loss, as in backwards pass, but don't update
                # model
                lossTarget = self.getTorchLossTarget(sampleInMap, sampleOutMap)
                sampleLosses.append(lossTarget.item())

            losses.append(sampleLosses)

        # returning: list with loss scalar per sample
        return losses

    def infer(self, inMap):
        """
        Does inference (i.e. forward pass only)
        """
        self.module.eval()

        # if batchesPerStep is 1, a dimension will be missing
        if self.dataFeed.batchesPerStep() == 1:
            inMap = _add_dimension(inMap)

        # perform forwards pass for each substep
        stepOutMap = {}
        for outName in self.outNames:
            stepOutMap[outName] = []

        for substepi in range(self.dataFeed.batchesPerStep()):

            substepTorchInputs = [
                torch.Tensor(inMap[inId][substepi][0:])
                for inId in inMap.keys()
            ]

            # forward pass
            substepOutputs = self.module(substepTorchInputs)
            substepOutputTensors = substepOutputs.detach()

            if len(self.outNames) == 1:
                stepOutMap[self.outNames[0]].append(
                    substepOutputTensors.numpy())
            else:
                for j, outName in enumerate(self.outNames):
                    npTensor = substepOutputTensors[j].numpy()
                    stepOutMap[outName].append(npTensor)

        # returning: list with one entry per substep, each containing
        # one entry per sample
        return stepOutMap


# add an extra dimension to each item in the input dict
def _add_dimension(inMap):
    return {k: np.expand_dims(v, axis=0) for k, v in inMap.items()}
