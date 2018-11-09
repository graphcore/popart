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
                 earlyInfo, module):
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
            earlyInfo=earlyInfo,
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

        inputDataInfos = [self.earlyInfo.get(tid) for tid in self.inNames]
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

    def step(self, inMap):
        """
        TODO : sort out case of tuples of tuples for outputs

        w.r.t. parameter names. It is possible to flatten
        named_parameters of module, but it is not clear that
        the order will always correspond to the order of the
        onnx "trace", so I won't.
        """
        torchOptimizer = self.getTorchOptimizer()
        self.module.train()
        batchSize = self.dataFeed.samplesPerBatch()

        # perform forwards - backwards - update
        # for each of the substeps (substep = batch)

        # this list we store a map of the output tensors
        # for each of the batches in the step
        substepOutMaps = []
        for substep in range(self.dataFeed.batchesPerStep()):

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

            substepOutMaps.append(substepOutMap)

        # returning: list with one entry per substep, of the
        # output of the batch processed at the substep
        return substepOutMaps
