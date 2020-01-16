import popart
from popart_core import _InferenceSessionCore, _TrainingSessionCore
from popart.session import _initAnchorArrays
import torch
import torch.nn as nn
from torch.onnx.utils import _model_to_graph
from torch.onnx import ONNX_ARCHIVE_MODEL_PROTO_NAME, ExportTypes, OperatorExportTypes
# Import all of torchwriter into this namespace
from .torchwriter import *

POPART_TYPE_MAP = {
    'torch.DoubleTensor': "FLOAT",
    'torch.FloatTensor': "FLOAT",
    'torch.HalfTensor': "FLOAT16",
    'torch.LongTensor': "INT32",
    'torch.IntTensor': "INT32",
    'torch.ShortTensor': "INT16",
    'torch.CharTensor': "INT8"
}


def torch_to_popart_type(torch_type):
    try:
        return POPART_TYPE_MAP[torch_type]
    except ValueError:
        raise ValueError(f"Unsupported type {torch_type}")


class InferenceSession(_InferenceSessionCore):
    def __init__(self,
                 torchModel,
                 inputs,
                 targets,
                 losses,
                 deviceInfo,
                 batch_size=1,
                 batches_per_step=1,
                 inputShapeInfo=popart.InputShapeInfo(),
                 passes=popart.Patterns(),
                 userOptions=popart.SessionOptions()):

        self.torchModel = torchModel
        self.batch_size = batch_size
        self.losses = losses
        self.deviceInfo = deviceInfo
        self.batches_per_step = batches_per_step
        self.inputShapeInfo = inputShapeInfo
        self.anchor_returns = {}

        self.inputs = tuple()
        self.targets = tuple()

        if isinstance(inputs, torch.Tensor):
            inputs = (inputs, )
        if isinstance(targets, torch.Tensor):
            targets = (targets, )

        for tensor in inputs:
            if (tensor.shape[0] !=
                (self.batch_size * self.batches_per_step)) and (self.batch_size
                                                                != 1):
                raise RuntimeError(
                    f"Shape discrepancy in input tensor {tensor}, shape {tensor.shape}."
                    + "Dim 0 should be equal to :" +
                    f"batch size {self.batch_size} * bps {self.batches_per_step}"
                )
            reshape = tensor.view(batches_per_step, batch_size,
                                  *list(tensor.shape[1:]))
            self.inputs = self.inputs + (reshape[0, :], )
        for tensor in targets:
            reshape = tensor.view(batches_per_step, batch_size)
            self.targets = self.targets + (reshape[0, :], )

        self.outputs = self.torchModel(*self.inputs)

        self.inputNames = [f"input_{i}" for i in range(len(self.inputs))]
        if isinstance(self.outputs, torch.Tensor):
            num_outputs = 1
        else:
            num_outputs = len(self.outputs)
        self.outputNames = [f"output_{i}" for i in range(num_outputs)]

        proto = self.createProto()

        losses = []
        for idx, (out, tgt) in enumerate(zip(self.outputNames, self.targets)):
            self.inputShapeInfo.add(
                f"target_{idx}",
                popart.TensorInfo(torch_to_popart_type(tgt.type()),
                                  list(tgt.shape)))

            losses.append(
                self.createLosses(self.losses, out, f"target_{idx}",
                                  f"loss_{idx}"))
            self.anchor_returns[out] = popart.AnchorReturnType("ALL")
            self.anchor_returns[f"loss_{idx}"] = popart.AnchorReturnType("ALL")

        if passes is None:
            passes = popart.Patterns()

        self.dataFeed = self.createDataFeed()

        super(InferenceSession,
              self).__init__(proto, self.dataFeed, self.deviceInfo, losses,
                             self.inputShapeInfo, userOptions, passes)

        self.replicationFactor = userOptions.replicatedGraphCount if \
            userOptions.enableReplicatedGraphs else 1
        self.accumulationFactor = userOptions.accumulationFactor if \
            userOptions.enableGradientAccumulation else 1

    def createLosses(self, loss, output, label=None, name="loss"):
        self.inputNames.append(label)
        if label:
            self.anchor_returns[label] = popart.AnchorReturnType("ALL")
        if isinstance(loss, torch.nn.NLLLoss):
            return popart.NllLoss(output, label, name)
        elif isinstance(loss, torch.nn.L1Loss):
            return popart.L1Loss(output, name)
        elif isinstance(loss, torch.nn.Identity):
            return popart.IdentityLoss(output, name)
        elif isinstance(loss, torch.Tensor):
            return popart.IdentityLoss(output, name)
        else:
            raise RuntimeError(
                "Only NLLLoss, L1Loss or Identity loss with manual "
                "loss construction are currently supported.")

    def createDataFeed(self):
        return popart.DataFlow(self.batches_per_step, self.anchor_returns)

    def createProto(self):

        graph, params_dict, _ = _model_to_graph(self.torchModel,
                                                args=self.inputs,
                                                training=True,
                                                verbose=True,
                                                input_names=self.inputNames,
                                                output_names=self.outputNames,
                                                _retain_param_name=True,
                                                do_constant_folding=True)
        proto, _ = graph._export_onnx(initializers=params_dict,
                                      dynamic_axes={},
                                      onnx_opset_version=9,
                                      defer_weight_export=False,
                                      strip_doc_string=True,
                                      keep_initializers_as_inputs=False)

        return proto

    def initAnchorArrays(self):
        self.anchorArrays = _initAnchorArrays(self)
        return self.anchorArrays

    def prepareDevice(self):
        err = popart.PrepareDeviceError()
        super(InferenceSession, self).prepareDevice(err)

        # If an error occurred during the perpareDevice raise an exception
        if not err.isSuccessful():
            raise popart.PrepareDeviceException(err)

    def run(self, *args):
        args = [a.detach().numpy() for a in args]
        step_data = dict(zip(self.inputNames, args))
        stepio = popart.PyStepIO(step_data, self.anchorArrays)
        super(InferenceSession, self).run(stepio)


class TrainingSession(_TrainingSessionCore):
    def __init__(self,
                 torchModel,
                 inputs,
                 targets,
                 optimizer,
                 losses,
                 deviceInfo,
                 batch_size=1,
                 batches_per_step=1,
                 inputShapeInfo=popart.InputShapeInfo(),
                 passes=popart.Patterns(),
                 userOptions=popart.SessionOptions()):

        self.torchModel = torchModel
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.losses = losses
        self.deviceInfo = deviceInfo
        self.batches_per_step = batches_per_step
        self.inputShapeInfo = inputShapeInfo
        self.anchor_returns = {}

        self.inputs = tuple()
        self.targets = tuple()

        if isinstance(inputs, torch.Tensor):
            inputs = (inputs, )
        if isinstance(targets, torch.Tensor):
            targets = (targets, )

        for tensor in inputs:
            if (tensor.shape[0] !=
                (self.batch_size * self.batches_per_step)) and (self.batch_size
                                                                != 1):
                raise RuntimeError(
                    f"Shape discrepancy in input tensor {tensor}, shape {tensor.shape}."
                    + "Dim 0 should be equal to :" +
                    f"batch size {self.batch_size} * bps {self.batches_per_step}"
                )
            reshape = tensor.view(batches_per_step, batch_size,
                                  *list(tensor.shape[1:]))
            self.inputs = self.inputs + (reshape[0, :], )
        for tensor in targets:
            reshape = tensor.view(batches_per_step, batch_size)
            self.targets = self.targets + (reshape[0, :], )

        self.outputs = self.torchModel(*self.inputs)

        self.inputNames = [f"input_{i}" for i in range(len(self.inputs))]
        if isinstance(self.outputs, torch.Tensor):
            num_outputs = 1
        else:
            num_outputs = len(self.outputs)
        self.outputNames = [f"output_{i}" for i in range(num_outputs)]

        proto = self.createProto()

        losses = []
        for idx, (out, tgt) in enumerate(zip(self.outputNames, self.targets)):
            self.inputShapeInfo.add(
                f"target_{idx}",
                popart.TensorInfo(torch_to_popart_type(tgt.type()),
                                  list(tgt.shape)))

            losses.append(
                self.createLosses(self.losses, out, f"target_{idx}",
                                  f"loss_{idx}"))
            self.anchor_returns[out] = popart.AnchorReturnType("ALL")
            self.anchor_returns[f"loss_{idx}"] = popart.AnchorReturnType("ALL")

        if passes is None:
            passes = popart.Patterns()

        self.dataFeed = self.createDataFeed()

        super(TrainingSession,
              self).__init__(proto, self.dataFeed, losses,
                             self.createOptimizer(), self.deviceInfo,
                             self.inputShapeInfo, userOptions, passes)

        self.replicationFactor = userOptions.replicatedGraphCount if \
            userOptions.enableReplicatedGraphs else 1
        self.accumulationFactor = userOptions.accumulationFactor if \
            userOptions.enableGradientAccumulation else 1

    def createLosses(self, loss, output, label=None, name="loss"):
        self.inputNames.append(label)
        if label:
            self.anchor_returns[label] = popart.AnchorReturnType("ALL")
        if isinstance(loss, torch.nn.NLLLoss):
            return popart.NllLoss(output, label, name)
        elif isinstance(loss, torch.nn.L1Loss):
            return popart.L1Loss(output, name)
        elif isinstance(loss, torch.nn.Identity):
            return popart.IdentityLoss(output, name)
        elif isinstance(loss, torch.Tensor):
            return popart.IdentityLoss(output, name)
        else:
            raise RuntimeError(
                "Only NLLLoss, L1Loss or Identity loss with manual "
                "loss construction are currently supported.")

    def createDataFeed(self):
        return popart.DataFlow(self.batches_per_step, self.anchor_returns)

    def createProto(self):

        graph, params_dict, _ = _model_to_graph(self.torchModel,
                                                args=self.inputs,
                                                training=True,
                                                verbose=True,
                                                input_names=self.inputNames,
                                                output_names=self.outputNames,
                                                _retain_param_name=True,
                                                do_constant_folding=True)
        proto, _ = graph._export_onnx(initializers=params_dict,
                                      dynamic_axes={},
                                      onnx_opset_version=9,
                                      defer_weight_export=False,
                                      strip_doc_string=True,
                                      keep_initializers_as_inputs=False)

        return proto

    def createOptimizer(self):
        if not isinstance(self.optimizer, torch.optim.SGD):
            raise RuntimeError("PopART currently only accepts SGD optimizers.")
        elif self.optimizer.defaults["nesterov"]:
            raise RuntimeError("Nesterov momentum is currently not supported.")
        return popart.SGD({
            "defaultLearningRate": (self.optimizer.defaults["lr"], False),
            "defaultMomentum": (self.optimizer.defaults["momentum"], False),
            "defaultWeightDecay":
            (self.optimizer.defaults["weight_decay"], False),
            "defaultDampening": (self.optimizer.defaults["dampening"], False)
        })

    def initAnchorArrays(self):
        self.anchorArrays = _initAnchorArrays(self)
        return self.anchorArrays

    def prepareDevice(self):
        err = popart.PrepareDeviceError()
        super(TrainingSession, self).prepareDevice(err)

        # If an error occurred during the perpareDevice raise an exception
        if not err.isSuccessful():
            raise popart.PrepareDeviceException(err)

    def run(self, *args):
        args = [a.detach().numpy() for a in args]
        step_data = dict(zip(self.inputNames, args))
        stepio = popart.PyStepIO(step_data, self.anchorArrays)
        super(TrainingSession, self).run(stepio)
