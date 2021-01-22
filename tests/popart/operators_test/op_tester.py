# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import numpy as np
import pytest
import popart
import onnx
import torch

# `import test_util` requires adding to sys.path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu


# Usage:
#   Add `op_tester` as an argument to a test function
#   In the test function:
#       Create a function to initialize the `Builder`.
#       This function should return a list of anchors.
#
#       Create a function to produce reference output.
#       This should return a list of reference values
#       the indices of which should correspond to the
#       anchors they reference.
#       The list of references must be the same length
#       as the list of anchors.
#       To exclude an anchor from testing, use `None`
#       at the anchors index in the refernece list.
#
#       The reference function takes one argument, `ref_data`.
#       `ref_data.getOutputTensorGrad(i)`, will return the gradient
#       of the output tensor at index `i`.
#
#   Call op_tester.run(init_builder, reference)
@pytest.fixture
def op_tester(tmpdir):
    class Builder:
        def __init__(self, opsets=None, check_model=True):
            self._builder = popart.Builder(opsets=opsets)

            self._input_map = {}
            self._init_input_map = {}
            self._outputs = []
            self._check_model = check_model

        def addInputTensor(self, data, debugName=""):
            shape = popart.TensorInfo(data)

            tensor_id = self._builder.addInputTensor(shape, debugName)

            self._input_map[tensor_id] = data

            return tensor_id

        def addInitializedInputTensor(self, data):
            tensor_id = self._builder.addInitializedInputTensor(data)
            self._init_input_map[tensor_id] = data

            return tensor_id

        def addOutputTensor(self, tensorId):
            self._outputs.append(tensorId)
            self._builder.addOutputTensor(tensorId)

        def getModelProto(self):
            proto = self._builder.getModelProto()

            # TODO (T33079): decide if we can apply this check unconditionally
            # in the underlying builder implementation
            if self._check_model:
                print("Checking model proto with onnx")
                model = onnx.load_from_string(proto)
                onnx.checker.check_model(model)

            return proto

        def __getattr__(self, attr):
            return getattr(self._builder, attr)

    class RefData:
        def __init__(self, outputs, anchor_map):
            self._outputs = outputs
            self._anchor_map = anchor_map

        def getOutputTensorGrad(self, index):
            tensorId = self._outputs[index]
            gradId = popart.reservedGradientPrefix() + tensorId
            return self._anchor_map[gradId]

        def getOutputTensor(self, index):
            tensorId = self._outputs[index]
            return self._anchor_map[tensorId]

    class OpTester:
        def __init__(self, logging_dir):
            np.random.seed(0)
            self.patterns = popart.Patterns(popart.PatternsLevel.Minimal)
            self.options = popart.SessionOptions()
            self.options.opxAliasChecking = True
            self.options.opxModifyChecking = True
            self.logging_dir = logging_dir
            self.numIPUs = 1
            self.tilesPerIPU = None
            self.rtol = 1e-05
            self.atol = 1e-08
            self.check_shapes = True
            self.check_dtypes = True
            self.equal_nan = False
            self.inplacing = True
            self.lossReduction = popart.ReductionType.Mean

            # TODO (T33079): This can be removed when all tests pass model validation.
            self.check_model = False

        def verifyTensor(self, t1, ref):
            if self.check_shapes:
                if t1.shape != ref.shape:
                    print('shape mismatch {} != {}'.format(
                        t1.shape, ref.shape))
                assert t1.shape == ref.shape

            if self.check_dtypes:
                if t1.dtype != ref.dtype:
                    print('dtype mismatch {} != {}'.format(
                        t1.dtype, ref.dtype))
                assert t1.dtype == ref.dtype

            if not np.allclose(t1, ref, self.rtol, self.atol, self.equal_nan):
                print('rtol:{} atol:{}'.format(self.rtol, self.atol))
                print('Popart:\n{}'.format(t1))
                print('Torch:\n{}'.format(ref))
                print('Diff:\n{}'.format(np.subtract(t1, ref)))
                isclose = np.isclose(t1, ref, self.rtol, self.atol,
                                     self.equal_nan)
                print('IsClose:\n{}'.format(isclose))
                indices = np.argwhere(np.logical_not(isclose))
                print('# not close:', indices.shape[0])
                for i in indices[0:10]:
                    print(i, 'Popart:', t1[tuple(i)], 'Torch:', ref[tuple(i)])

            assert np.allclose(t1, ref, self.rtol, self.atol, self.equal_nan)

        def setPatterns(self, patterns, enableRuntimeAsserts=None):
            """ Accept either a Patterns instance, a list of strings or a PatternsLevel. If enableRuntimeAsserts 
                is set you will get an error if you don't enable all mandatory patterns.
            """
            if isinstance(patterns, popart.Patterns):
                self.patterns = patterns
            else:
                self.patterns = popart.Patterns(patterns)

            if enableRuntimeAsserts is not None:
                self.patterns.enableRuntimeAsserts(enableRuntimeAsserts)

        def run(self,
                init_builder,
                reference,
                step_type='infer',
                opsets=None,
                optimizer=popart.ConstSGD(0.01),
                seed=None):
            assert step_type in ('infer', 'train')

            bld = Builder(opsets=opsets, check_model=self.check_model)

            anchors = {}

            # Allows to pass additional arguments to init_builder, if required
            # by the specific init_builder function implementation.
            kwargs = {}
            kwargs = tu.filter_dict(kwargs, init_builder)
            anchorIds = init_builder(bld, **kwargs)

            for anchorId in anchorIds:
                if anchorId not in bld._init_input_map:
                    anchors[anchorId] = popart.AnchorReturnType("All")

            dataFlow = popart.DataFlow(1, anchors)

            self.options.logDir = self.logging_dir

            if self.tilesPerIPU is not None:
                device = tu.create_test_device(numIpus=self.numIPUs,
                                               tilesPerIPU=self.tilesPerIPU)
                print(f"Created device {device} with {self.numIPUs}"
                      f" IPUs and {self.tilesPerIPU} tiles per IPU")
            else:
                device = tu.create_test_device(numIpus=self.numIPUs)
                print(f"Created device {device} with {self.numIPUs} IPUs")

            self.patterns.InPlace = self.inplacing
            if step_type == 'infer':
                session = popart.InferenceSession(fnModel=bld.getModelProto(),
                                                  dataFlow=dataFlow,
                                                  deviceInfo=device,
                                                  patterns=self.patterns,
                                                  userOptions=self.options)
            else:
                assert step_type == 'train'
                # Apply reduction to output (assumed to be the
                # first anchorId) to ensure it is scalar
                lossId = anchorIds[0]
                lossId = bld.aiGraphcore.identityloss(
                    [lossId], reduction=self.lossReduction)

                session = popart.TrainingSession(fnModel=bld.getModelProto(),
                                                 dataFlow=dataFlow,
                                                 loss=lossId,
                                                 optimizer=optimizer,
                                                 deviceInfo=device,
                                                 patterns=self.patterns,
                                                 userOptions=self.options)

            anchor_map = session.initAnchorArrays()

            session.prepareDevice()

            if seed is not None:
                session.setRandomSeed(seed)

            for k, v in bld._input_map.items():
                if not v.flags['C_CONTIGUOUS']:
                    # need to call np.ascontiguousarray
                    # `x = np.ascontiguousarray(x)`
                    raise Exception(
                        'Input "{}" to popart.PyStepIO is not C_CONTIGUOS'.
                        format(k))

            # Add the replication dimension to the inputs
            inputs = {}
            for k, v in bld._input_map.items():
                if self.options.replicatedGraphCount > 1:
                    um = (self.options.replicatedGraphCount, )
                    um = um + tuple([1] * np.ndim(v))

                    # we add this offset to ensure that samples on devices are distinct
                    offset = 1 * np.arange(
                        self.options.replicatedGraphCount).astype(
                            v.dtype).reshape(um)

                    inputs[k] = np.tile(v, um) + offset

                else:
                    inputs[k] = v

            stepio = popart.PyStepIO(inputs, anchor_map)

            if (step_type == 'train'):
                session.weightsFromHost()

            session.run(stepio)

            if (step_type == 'train'):
                session.weightsToHost()

            ref_out = reference(RefData(bld._outputs, anchor_map))

            def fix_type(t):
                if isinstance(t, torch.Tensor):
                    return t.data.numpy()
                elif isinstance(t, np.ndarray):
                    return t
                elif isinstance(t, np.float32):
                    return t
                elif isinstance(t, np.float16):
                    return t
                elif t is None:
                    return None
                else:
                    raise Exception('unexpected type', type(t))

            ref_out = [fix_type(i) for i in ref_out]
            for index, key in enumerate(anchorIds):
                if key in anchors:
                    if ref_out[index] is not None:
                        print('Testing anchor "{}"...'.format(key))
                        self.verifyTensor(anchor_map[key], ref_out[index])
                    else:
                        print('Not Testing anchor "{}" as it is None'.format(
                            key))
                elif key in bld._init_input_map:
                    if ref_out[index] is not None:
                        print('Testing weight "{}"...'.format(key))
                        weightInfo = session.getInfo(key)
                        print('Weight info shape:{} type:{}',
                              weightInfo.shape(), weightInfo.data_type_lcase())
                        weights = {}
                        weights[key] = np.empty(
                            shape=weightInfo.shape(),
                            dtype=weightInfo.data_type_lcase())
                        weightsIo = popart.PyWeightsIO(weights)
                        session.readWeights(weightsIo)

                        self.verifyTensor(weights[key], ref_out[index])

                    else:
                        print('Not Testing weight "{}" as it is None'.format(
                            key))

            return session

    return OpTester(str(tmpdir))
