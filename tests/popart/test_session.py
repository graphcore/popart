import numpy as np
import popart
import test_util as tu


class TestSession:
    def __init__(self):
        self.options = popart.SessionOptions()
        self.device = 'cpu'
        self.numIPUs = 1
        self.mode = 'inference'
        self.passes = None
        self.batchesPerStep = 1

    def prepare_and_run(self, init_builder, ins=None):
        self.prepare(init_builder)
        return self.run(ins)

    def prepare(self, init_builder):
        self._builder = _Builder()
        anchorIds = init_builder(self._builder)
        self._builder._check_inputs()
        anchors = _get_anchors(anchorIds, self._builder)

        dataFlow = popart.DataFlow(self.batchesPerStep, anchors)
        proto = self._builder.getModelProto()
        losses = self._get_losses(anchorIds)
        device = self._get_device()

        optimizer = popart.ConstSGD(0.01)

        self._session = self._get_session(
            fnModel=proto,
            dataFeed=dataFlow,
            losses=losses,
            optimizer=optimizer,
            deviceInfo=device,
            passes=self.passes,
            userOptions=self.options)
        self._device_prepared = False

    def run(self, ins=None):
        self._check_device_prepared()
        inputs = self._get_inputs(ins)

        # use a new anchor map each time to prevent another
        # call to run overwriting previous results
        _anchor_map = self._session.initAnchorArrays()
        stepio = popart.PyStepIO(inputs, _anchor_map)
        self._session.run(stepio)
        return _anchor_map

    def _get_losses(self, anchorIds):
        if self._builder._losses:
            print(f'Returning losses from builder {self._builder._losses}')
            return self._builder._losses
        else:
            print(f'Returning default losses')
            return [
                popart.L1Loss(anchorIds[0], "l1LossVal", 0.1,
                              popart.ReductionType.Sum)
            ]

    def _get_session(self, **kwargs):
        def create_session(valid_args, session_type):
            session_args = {}
            for k, v in kwargs.items():
                if k in valid_args:
                    session_args[k] = v
            return session_type(**session_args)

        if self.mode == 'inference':
            return create_session(('fnModel', 'dataFeed', 'losses',
                                   'deviceInfo', 'passes', 'userOptions'),
                                  popart.InferenceSession)
        elif self.mode == 'train':
            return create_session(
                ('fnModel', 'dataFeed', 'losses', 'optimizer', 'deviceInfo',
                 'passes', 'userOptions'), popart.TrainingSession)

    def _get_device(self):
        if self.device == "cpu":
            return tu.get_poplar_cpu_device()
        elif self.device == "ipu_model":
            return tu.get_ipu_model(numIPUs=self.numIPUs)
        else:
            return self.device

    def _get_inputs(self, ins):
        inputs = {}
        for k, v in self._builder._get_inputs().items():
            if ins and k in ins:
                inputs[k] = ins[k]
            else:
                if self.batchesPerStep == 1:
                    inputs[k] = v
                else:
                    inputs[k] = np.stack(
                        [v for _ in range(self.batchesPerStep)])

        if ins:
            for k in ins.keys():
                if k not in inputs:
                    raise KeyError(
                        f'supplied input "{k}" is not a valid input id')

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
        self._losses = []

    def addInputTensor(self, data, debug_prefix=None):
        shape = popart.TensorInfo(data)

        if debug_prefix:
            tensor_id = self._builder.addInputTensor(shape, debug_prefix)
        else:
            tensor_id = self._builder.addInputTensor(shape)

        self._input_map[tensor_id] = data

        return tensor_id

    def addInitializedInputTensor(self, data):
        shape = popart.TensorInfo(data)

        tensor_id = self._builder.addInitializedInputTensor(data)
        self._init_input_map[tensor_id] = data

        return tensor_id

    def addL1Loss(self, *args):
        self._losses.append(popart.L1Loss(*args))
        return self._losses[-1]

    def addOutputTensor(self, tensorId):
        self._outputs.append(tensorId)
        self._builder.addOutputTensor(tensorId)

    def __getattr__(self, attr):
        return getattr(self._builder, attr)

    def _check_inputs(self):
        for k, v in self._input_map.items():
            if not v.flags['C_CONTIGUOUS']:
                # need to call np.ascontiguousarray
                # `x = np.ascontiguousarray(x)`
                raise Exception(
                    'Input "{}" to popart.PyStepIO is not C_CONTIGUOS'.format(
                        k))

    def _get_inputs(self):
        return {k: v for k, v in self._input_map.items()}


def _get_anchors(anchorIds, builder):
    anchors = {}
    for anchorId in anchorIds:
        if anchorId not in builder._init_input_map:
            anchors[anchorId] = popart.AnchorReturnType('ALL')

    return anchors
