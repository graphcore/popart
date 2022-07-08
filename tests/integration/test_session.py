# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import numpy as np
import popart


class PopartTestSession:
    def __init__(self):
        self.options = popart.SessionOptions()
        self.device = None
        self.numIPUs = 1
        self.mode = "inference"
        self.patterns = None
        self.batchesPerStep = 1

    def prepare_and_run(self, init_builder, ins=None, device=None):
        self.prepare(init_builder, device=device)
        return self.run(ins)

    def prepare(self, init_builder, device=None):
        self._builder = _Builder()
        anchorIds = init_builder(self._builder)
        self._builder._check_inputs()
        anchors = _get_anchors(anchorIds, self._builder)

        dataFlow = popart.DataFlow(self.batchesPerStep, anchors)
        proto = self._builder.getModelProto()
        loss = self._get_loss(anchorIds)

        optimizer = popart.ConstSGD(0.01)

        self._session = self._get_session(
            fnModel=proto,
            dataFlow=dataFlow,
            loss=loss,
            optimizer=optimizer,
            deviceInfo=device,
            patterns=self.patterns,
            userOptions=self.options,
        )
        self._device_prepared = False

    def run(self, ins=None):
        self._check_device_prepared()
        inputs = self._get_inputs(ins)

        # use a new anchor map each time to prevent another
        # call to run overwriting previous results
        _anchor_map = self._session.initAnchorArrays()
        stepio = popart.PyStepIO(inputs, _anchor_map)
        self._session.weightsFromHost()
        self._session.run(stepio)
        return _anchor_map

    def _get_loss(self, anchorIds):
        if self._builder._loss:
            print(f"Returning loss from builder {self._builder._loss}")
            return self._builder._loss
        else:
            print("Returning default loss")
            return anchorIds[0]

    def _get_session(self, **kwargs):
        def create_session(valid_args, session_type):
            session_args = {}
            for k, v in kwargs.items():
                if k in valid_args:
                    session_args[k] = v
            return session_type(**session_args)

        if self.mode == "inference":
            return create_session(
                ("fnModel", "dataFlow", "deviceInfo", "patterns", "userOptions"),
                popart.InferenceSession,
            )
        elif self.mode == "train":
            return create_session(
                (
                    "fnModel",
                    "dataFlow",
                    "loss",
                    "optimizer",
                    "deviceInfo",
                    "patterns",
                    "userOptions",
                ),
                popart.TrainingSession,
            )

    def _get_inputs(self, ins):
        inputs = {}
        for k, v in self._builder._get_inputs().items():
            if ins and k in ins:
                inputs[k] = ins[k]
            else:
                if self.batchesPerStep == 1:
                    inputs[k] = v
                else:
                    inputs[k] = np.stack([v for _ in range(self.batchesPerStep)])

        if ins:
            for k in ins.keys():
                if k not in inputs:
                    raise KeyError(f'supplied input "{k}" is not a valid input id')

        return inputs

    def _check_device_prepared(self):
        if not self._device_prepared:
            self._session.prepareDevice()
            self._device_prepared = True


class _Builder:
    def __init__(self, opsets=None):
        self._builder = popart.Builder(opsets=opsets)

        self._input_map = {}
        self._init_input_map = {}
        self._outputs = []
        self._loss = []

    def addInputTensor(self, data, debug_prefix=None):
        shape = popart.TensorInfo(data)

        if debug_prefix:
            tensor_id = self._builder.addInputTensor(shape, debug_prefix)
        else:
            tensor_id = self._builder.addInputTensor(shape)

        self._input_map[tensor_id] = data

        return tensor_id

    def addInitializedInputTensor(self, data, debug_prefix=None):
        if debug_prefix:
            tensor_id = self._builder.addInitializedInputTensor(data, debug_prefix)
        else:
            tensor_id = self._builder.addInitializedInputTensor(data)

        self._init_input_map[tensor_id] = data

        return tensor_id

    def addOutputTensor(self, tensorId):
        self._outputs.append(tensorId)
        self._builder.addOutputTensor(tensorId)

    def setLoss(self, tensorId):
        self._loss = tensorId

    def __getattr__(self, attr):
        return getattr(self._builder, attr)

    def _check_inputs(self):
        for k, v in self._input_map.items():
            if not v.flags["C_CONTIGUOUS"]:
                # need to call np.ascontiguousarray
                # `x = np.ascontiguousarray(x)`
                raise Exception(
                    'Input "{}" to popart.PyStepIO is not C_CONTIGUOS'.format(k)
                )

    def _get_inputs(self):
        return {k: v for k, v in self._input_map.items()}


def _get_anchors(anchorIds, builder):
    anchors = {}
    for anchorId in anchorIds:
        if anchorId not in builder._init_input_map:
            anchors[anchorId] = popart.AnchorReturnType("All")

    return anchors
