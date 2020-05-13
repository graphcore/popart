# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import numpy as np
import pytest
import popart
import torch

# `import test_util` requires adding to sys.path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu


# Usage:
#   Add `ipu_op_tester` as an argument to a test function
#   In the test function:
#       Create a function to initialize the `Builder`.
#       This function should return a list of anchors.
#
#       Create a function to produce refernece output.
#       This should return a list of refernece values
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
#   Call ipu_op_tester.run(init_builder, reference)
@pytest.fixture
def ipu_op_tester(tmpdir):
    class Builder:
        def __init__(self, opsets=None):
            self._builder = popart.Builder(opsets=opsets)
            self._input_map = {}
            self._init_input_map = {}
            self._outputs = []

        def addInputTensor(self, data):
            shape = popart.TensorInfo(data)

            tensor_id = self._builder.addInputTensor(shape)
            self._input_map[tensor_id] = data

            return tensor_id

        def addInitializedInputTensor(self, data):
            shape = popart.TensorInfo(data)

            tensor_id = self._builder.addInitializedInputTensor(data)
            self._init_input_map[tensor_id] = data

            return tensor_id

        def addOutputTensor(self, tensorId):
            self._outputs.append(tensorId)
            self._builder.addOutputTensor(tensorId)

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

    class OpTester:
        def __init__(self, logging_dir):
            np.random.seed(0)
            self.passes = []
            self.logging_dir = logging_dir
            self.rtol = 1e-05
            self.atol = 1e-08
            self.check_shapes = True

        def run(self, init_builder, reference, step_type='infer', opsets=None):
            assert step_type in ('infer', 'train')

            popart.getLogger().setLevel("TRACE")

            bld = Builder(opsets=opsets)

            anchors = {}
            anchorIds = init_builder(bld)
            for anchorId in anchorIds:
                if anchorId not in bld._init_input_map:
                    anchors[anchorId] = popart.AnchorReturnType("All")

            dataFlow = popart.DataFlow(1, anchors)

            if (step_type == 'train'):
                optimizer = popart.ConstSGD(0.01)
            else:
                optimizer = None

            losses = [popart.IdentityLoss(anchorIds[0], "idLossVal")]
            proto = bld.getModelProto()

            opts = popart.SessionOptions()
            opts.logDir = self.logging_dir
            opts.virtualGraphMode = popart.VirtualGraphMode.Manual

            device = tu.create_test_device(numIpus=4)

            if (step_type == 'infer'):
                session = popart.InferenceSession(fnModel=proto,
                                                  dataFeed=dataFlow,
                                                  deviceInfo=device,
                                                  patterns=popart.Patterns(
                                                      self.passes),
                                                  userOptions=opts)
            elif (step_type == 'train'):
                session = popart.TrainingSession(fnModel=proto,
                                                 dataFeed=dataFlow,
                                                 losses=losses,
                                                 optimizer=optimizer,
                                                 deviceInfo=device,
                                                 patterns=popart.Patterns(
                                                     self.passes),
                                                 userOptions=opts)

            anchor_map = session.initAnchorArrays()

            session.prepareDevice()

            for k, v in bld._input_map.items():
                if not v.flags['C_CONTIGUOUS']:
                    # need to call np.ascontiguousarray
                    # `x = np.ascontiguousarray(x)`
                    raise Exception(
                        'Input "{}" to popart.PyStepIO is not C_CONTIGUOS'.
                        format(k))

            stepio = popart.PyStepIO(bld._input_map, anchor_map)

            if (step_type == 'train'):
                session.weightsFromHost()

            session.run(stepio)

            ref_out = reference(RefData(bld._outputs, anchor_map))

            def fix_type(t):
                if isinstance(t, torch.Tensor):
                    return t.data.numpy()
                elif isinstance(t, np.ndarray):
                    return t
                elif isinstance(t, np.float32):
                    return t
                elif t is None:
                    return None
                else:
                    raise Exception('unexpected type', type(t))

            ref_out = [fix_type(i) for i in ref_out]
            for index, key in enumerate(anchors):
                if ref_out[index] is not None:
                    print('Testing anchor "{}"...'.format(key))

                    if self.check_shapes:
                        if anchor_map[key].shape != ref_out[index].shape:
                            print('shape mismatch {} != {}'.format(
                                anchor_map[key].shape, ref_out[index].shape))
                        assert anchor_map[key].shape == ref_out[index].shape

                    if not np.allclose(anchor_map[key], ref_out[index],
                                       self.rtol, self.atol):
                        print('rtol:{} atol:{}'.format(self.rtol, self.atol))
                        print('Popart:\n{}'.format(anchor_map[key]))
                        print('Torch:\n{}'.format(ref_out[index]))
                        print('Diff:\n{}'.format(
                            np.subtract(anchor_map[key], ref_out[index])))
                        print('IsClose:\n{}'.format(
                            np.isclose(anchor_map[key], ref_out[index],
                                       self.rtol, self.atol)))

                    assert np.allclose(anchor_map[key], ref_out[index],
                                       self.rtol, self.atol)
                else:
                    print('Not Testing anchor "{}" as it is None'.format(key))

            return session

    return OpTester((tmpdir))
