# Copyright (c) 2018 Graphcore Ltd. All rights reserved.
import numpy as np
import torch
import torch.utils
import torch.utils.data
import popart
from popart.writer import NetWriter
from popart import ConstSGD, SGD
import onnx


def conv3x3(in_planes, out_planes, stride=1):
    """Create 3x3 convolution with padding."""
    return torch.nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class PytorchNetWriter(NetWriter):
    def __init__(
        self,
        inNames,
        outNames,
        optimizer,
        dataFlow,
        inputShapeInfo,
        module,
        samplesPerBatch,
    ):
        """
        Create a PytorchNetWriter class.

        module:
          -- pytorch module (whose forward does not have the loss layers)
        all others:
          -- parameters passed to base class.
        """
        NetWriter.__init__(
            self,
            inNames=inNames,
            outNames=outNames,
            optimizer=optimizer,
            inputShapeInfo=inputShapeInfo,
            dataFlow=dataFlow,
        )

        self.module = module
        self.samplesPerBatch = samplesPerBatch

    def getTorchOptimizer(self):
        """
        Convert PopART's Optimizer to a torch Optimizer.
        """
        if isinstance(self.optimizer, SGD) or isinstance(self.optimizer, ConstSGD):
            return torch.optim.SGD(
                self.module.parameters(),
                lr=self.optimizer.learningRates().getDefault().val(),
                weight_decay=self.optimizer.weightDecays().getDefault().val(),
                momentum=self.optimizer.momentums().getDefault().val(),
            )
        else:
            raise RuntimeError("unrecognised optimizer")

    def saveModel(self, fnModel):
        print("Writing ONNX model to protobuf file %s" % (fnModel,))
        # jump into eval mode, just to write the onnx model.
        # note that this might do strange things with batch-normalisation (?)
        self.module.eval()

        inputDataInfos = [self.inputShapeInfo.get(tid) for tid in self.inNames]
        inputData = []
        containsint64 = False
        for info in inputDataInfos:
            shape = info.shape()
            dt = info.data_type_lcase()
            if dt == "int32":
                dt = "int64"  # torch labels must be 'long'
                containsint64 = True
            inputData.append(torch.from_numpy(np.ones(shape=shape, dtype=dt)))

        torch.onnx.export(
            self.module,
            inputData,
            fnModel,
            verbose=False,
            input_names=self.inNames,
            output_names=self.outNames,
        )

        # If the model contains 'long' tensors (e.g. in case of exporting
        # nllloss), they must be converted to int32
        # Note: in models with reshape ops, the 'shape' tensor will be converted
        # to int32 by the blanket conversion. This leads to a technically invalid
        # onnx model. So we only convert when we know we definitely have int64 tensors.
        if containsint64:
            graph_transformer = popart.GraphTransformer(fnModel)
            graph_transformer.convertINT64ToINT32()
            graph_transformer.convertAllFixedPointInitializersToConstants()
            proto = graph_transformer.getModelProto()
            popart.Builder(proto).saveModelProto(fnModel)

        onnx.checker.check_model(fnModel)

    def train(self, inMap):
        """
        Run training.
        """
        torchOptimizer = self.getTorchOptimizer()
        self.module.train()

        # if batchesPerStep is 1, a dimension will be missing
        if self.dataFlow.batchesPerStep() == 1:
            inMap = _add_dimension(inMap)

        # perform forwards - backwards - update
        # for each of the substeps (substep = batch)

        stepParameterMap = []
        for substepi in range(self.dataFlow.batchesPerStep()):

            substepParameterMap = {}
            substepInMap = {}
            for inId in inMap.keys():
                substepInMap[inId] = inMap[inId][substepi][0:]

            torchOptimizer.zero_grad()
            substepInputs = []
            for name in self.inNames:
                dt = self.inputShapeInfo.get(name).data_type_lcase()
                if dt == "int32":
                    dt = "int64"  # torch labels must be 'long'
                substepInputs.append(torch.tensor(substepInMap[name].astype(dt)))

            # forward pass
            substepOutput = self.module(substepInputs)

            if len(self.outNames) != 1:
                raise RuntimeError("Expecting single scalar loss")

            # backwards pass
            substepOutput.backward()
            torchOptimizer.step()

            for name, param in self.module.named_parameters():
                substepParameterMap[name] = param.data
            stepParameterMap.append(substepParameterMap)

        # returning: list with one entry per substep, of the
        # parameters of the model processed at the substep
        return stepParameterMap

    def infer(self, inMap):
        """
        Run inference (i.e. forward pass only).
        """
        self.module.eval()

        # if batchesPerStep is 1, a dimension will be missing
        if self.dataFlow.batchesPerStep() == 1:
            inMap = _add_dimension(inMap)

        # perform forwards pass for each substep
        stepOutMap = {}
        for outName in self.outNames:
            stepOutMap[outName] = []

        for substepi in range(self.dataFlow.batchesPerStep()):

            substepTorchInputs = [
                torch.Tensor(inMap[inId][substepi][0:]) for inId in inMap.keys()
            ]

            # forward pass
            substepOutputs = self.module(substepTorchInputs)
            substepOutputTensors = substepOutputs.detach()

            if len(self.outNames) == 1:
                stepOutMap[self.outNames[0]].append(substepOutputTensors.numpy())
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
