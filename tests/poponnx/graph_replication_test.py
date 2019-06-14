import numpy as np
import pytest
import poponnx
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
#   Call op_tester.run(init_builder, reference)
@pytest.fixture
def op_tester(tmpdir):
    class Builder:
        def __init__(self, opsets=None):
            self._builder = poponnx.Builder(opsets=opsets)

            self._input_map = {}
            self._init_input_map = {}
            self._outputs = []

        def addInputTensor(self, data):
            shape = poponnx.TensorInfo(data)
            tensor_id = self._builder.addInputTensor(shape)
            self._input_map[tensor_id] = data

            return tensor_id

        def addInitializedInputTensor(self, data):
            shape = poponnx.TensorInfo(data)

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
            gradId = poponnx.reservedGradientPrefix() + tensorId
            return self._anchor_map[gradId]

    class OpTester:
        def __init__(self, logging_dir):
            np.random.seed(0)
            self.passes = []
            self.options = poponnx.SessionOptionsCore()
            self.logging_dir = logging_dir
            self.device = "cpu"
            self.numIPUs = 2
            self.rtol = 1e-05
            self.atol = 1e-08
            self.check_shapes = True

        def verifyTensor(self, t1, ref):
            if self.check_shapes:
                if t1.shape != ref.shape:
                    print('shape mismatch {} != {}'.format(
                        t1.shape, ref.shape))
                assert t1.shape == ref.shape

            if not np.allclose(t1, ref, self.rtol, self.atol):
                print('rtol:{} atol:{}'.format(self.rtol, self.atol))
                print('Poponnx:\n{}'.format(t1))
                print('Torch:\n{}'.format(ref))
                print('Diff:\n{}'.format(np.subtract(t1, ref)))
                print('IsClose:\n{}'.format(
                    np.isclose(t1, ref, self.rtol, self.atol)))

            assert np.allclose(t1, ref, self.rtol, self.atol)

        def run(self, init_builder, reference, step_type='infer', opsets=None):
            assert step_type in ('infer', 'train')

            bld = Builder(opsets=opsets)

            anchors = {}
            anchorIds = init_builder(bld)
            for anchorId in anchorIds:

                if anchorId not in bld._init_input_map:
                    anchors[anchorId] = poponnx.AnchorReturnType("ALL")

            dataFlow = poponnx.DataFlow(1, anchors)

            if (step_type == 'train'):
                optimizer = poponnx.ConstSGD(0.01)
            else:
                optimizer = None

            losses = [
                poponnx.L1Loss(anchorIds[0], "l1LossVal", 0.1,
                               poponnx.ReductionType.Mean)
            ]
            proto = bld.getModelProto()

            self.options.logDir = self.logging_dir

            if self.device == "cpu":
                device = tu.get_poplar_cpu_device()
            elif self.device == "ipu_model":
                device = tu.get_ipu_model(numIPUs=self.numIPUs)

            if step_type == 'infer':
                session = poponnx.InferenceSession(
                    fnModel=proto,
                    dataFeed=dataFlow,
                    deviceInfo=device,
                    passes=poponnx.Patterns(self.passes),
                    userOptions=self.options)
            else:
                session = poponnx.TrainingSession(
                    fnModel=proto,
                    dataFeed=dataFlow,
                    losses=losses,
                    optimizer=optimizer,
                    deviceInfo=device,
                    passes=poponnx.Patterns(self.passes),
                    userOptions=self.options)

            anchor_map = session.initAnchorArrays()

            session.prepareDevice()

            for k, v in bld._input_map.items():
                if not v.flags['C_CONTIGUOUS']:
                    # need to call np.ascontiguousarray
                    # `x = np.ascontiguousarray(x)`
                    raise Exception(
                        'Input "{}" to poponnx.PyStepIO is not C_CONTIGUOS'.
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

            stepio = poponnx.PyStepIO(inputs, anchor_map)

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
                        weightsIo = poponnx.PyWeightsIO(weights)
                        session.readWeights(weightsIo)

                        self.verifyTensor(weights[key], ref_out[index])

                    else:
                        print('Not Testing weight "{}" as it is None'.format(
                            key))

            return session

    return OpTester(str(tmpdir))


def test_weight_update(op_tester):

    A = np.ones((2, 4)).astype(np.float32)
    B = np.ones((4, 6)).astype(np.float32)
    C = np.zeros((2, 6)).astype(np.float32)

    alpha = 1.0
    beta = 1.0
    transA = False
    transB = False

    def init_builder(builder):
        i1 = builder.addInputTensor(A)
        i2 = builder.addInitializedInputTensor(B)
        i3 = builder.addInitializedInputTensor(C)
        o = builder.aiOnnx.gemm([i1, i2, i3], alpha, beta, transA, transB)
        builder.addOutputTensor(o)

        return [o, poponnx.reservedGradientPrefix() + i2, i2, i3]

    def reference(ref_data):
        class Module(torch.nn.Module):
            def __init__(self):
                super(Module, self).__init__()

                self.B = torch.nn.Parameter(
                    torch.tensor(np.ones((4, 6)).astype(np.float32)),
                    requires_grad=True)
                self.C = torch.nn.Parameter(
                    torch.tensor(np.zeros((2, 6)).astype(np.float32)),
                    requires_grad=True)
                self.matmul = torch.matmul

            def forward(self, inputs):

                x = 1.0 * self.matmul(inputs[0], self.B) + 1.0 * self.C
                return x

        module = Module()

        module.train()

        optimizer = torch.optim.SGD(
            module.parameters(), lr=0.01, weight_decay=0.0, momentum=0.0)

        a = torch.tensor(A, requires_grad=True)
        b = torch.tensor(B, requires_grad=True)
        c = torch.tensor(C, requires_grad=True)

        optimizer.zero_grad()

        # forward
        o = module([a])

        loss = torch.nn.L1Loss(reduction="mean")
        target = torch.zeros(o.size())
        output = 0.1 * loss(o, target)
        output.backward()
        optimizer.step()

        return [o, b.grad, module.B.data, module.C.data]

    op_tester.passes = ['GemmDecomposition', 'PreUniRepl']
    op_tester.run(init_builder, reference, 'train')


def test_weight_update_replicated(op_tester):
    A = np.random.rand(2, 4).astype(np.float32)
    B = np.ones((4, 6)).astype(np.float32)
    C = np.random.rand(2, 6).astype(np.float32)

    alpha = np.random.random(1).astype(np.float32)[0]
    beta = np.random.random(1).astype(np.float32)[0]
    transA = False
    transB = False

    replicationFactor = 4

    def init_builder(builder):
        i1 = builder.addInputTensor(A)
        i2 = builder.addInitializedInputTensor(B)
        i3 = builder.addInitializedInputTensor(C)
        o = builder.aiOnnx.gemm([i1, i2, i3], alpha, beta, transA, transB)
        builder.addOutputTensor(o)

        return [
            o,
            poponnx.reservedGradientPrefix() + i2, i2,
            poponnx.reservedGradientPrefix() + i3, i3
        ]

    def reference(ref_data):
        class Module(torch.nn.Module):
            def __init__(self):
                super(Module, self).__init__()

                self.b = torch.tensor(B, requires_grad=True)
                self.c = torch.tensor(C, requires_grad=True)

                # Create the weight tensors for pytorch
                self.B = torch.nn.Parameter(self.b, requires_grad=True)
                self.C = torch.nn.Parameter(self.c, requires_grad=True)

                self.matmul = torch.matmul

            def forward(self, inputs):
                # Perform the GEMM operation
                x = alpha * self.matmul(inputs[0], self.B) + beta * self.C
                return x

        module = Module()

        module.train()

        optimizer = torch.optim.SGD(
            module.parameters(), lr=0.01, weight_decay=0.0, momentum=0.0)

        a = torch.tensor(A, requires_grad=True)

        optimizer.zero_grad()

        outputs = ()

        # graph with gradient accumlation i.e. only update the weights after x passes
        for n in range(replicationFactor):
            # adding n as offset, as op_tester expects
            o = module([a + n])
            outputs = outputs + (o, )
            loss = torch.nn.L1Loss(reduction="mean")
            target = torch.zeros(o.size())
            output = 0.1 * loss(o, target) / replicationFactor
            output.backward()

        # Update the weights
        optimizer.step()

        # Add dimension to each output so we can concatenate them
        outputs = tuple(map(lambda x: torch.unsqueeze(x, 0), outputs))

        return [
            torch.cat(outputs), module.b.grad, module.B.data, module.c.grad,
            module.C.data
        ]

    op_tester.passes = ['GemmDecomposition', 'PreUniRepl']
    op_tester.options.enableReplicatedGraphs = True
    op_tester.options.replicatedGraphCount = replicationFactor
    op_tester.device = "ipu_model"
    op_tester.numIPUs = replicationFactor
    op_tester.run(init_builder, reference, 'train')


def test_replication_infer(op_tester):

    # 2 samples per device
    A = np.random.rand(2, 7).astype(np.float32)
    B = np.random.rand(7, 6).astype(np.float32)
    C = np.random.rand(1, 6).astype(np.float32)

    alpha = 1.0
    beta = 1.0
    transA = False
    transB = False

    replicationFactor = 4

    def init_builder(builder):
        i1 = builder.addInputTensor(A)
        i2 = builder.addInitializedInputTensor(B)
        i3 = builder.addInitializedInputTensor(C)
        o = builder.aiOnnx.gemm([i1, i2, i3], alpha, beta, transA, transB)
        builder.addOutputTensor(o)

        return [o]

    def reference(ref_data):
        class Module(torch.nn.Module):
            def __init__(self):
                super(Module, self).__init__()

                self.B = torch.nn.Parameter(torch.tensor(B))
                self.C = torch.nn.Parameter(torch.tensor(C))
                self.matmul = torch.matmul

            def forward(self, inputs):

                x = 1.0 * self.matmul(inputs[0], self.B) + 1.0 * self.C
                return x

        module = Module()

        module.eval()

        a = torch.tensor(A)
        b = torch.tensor(B)
        c = torch.tensor(C)

        # forward
        # Run the pytorch module multiple times to simulate the same
        # behaviour as poponnx. The offsets (with corresponding offsets
        # in op_tester) ensure that the samples are distinct between replicas
        o1 = module([a + 0.])
        o2 = module([a + 1.])
        o3 = module([a + 2.])
        o4 = module([a + 3.])

        return [
            torch.cat((torch.unsqueeze(o1, 0), torch.unsqueeze(o2, 0),
                       torch.unsqueeze(o3, 0), torch.unsqueeze(o4, 0)))
        ]

    op_tester.passes = ['GemmDecomposition', 'PreUniRepl']
    op_tester.options.enableReplicatedGraphs = True
    op_tester.options.replicatedGraphCount = replicationFactor
    op_tester.device = "ipu_model"
    op_tester.numIPUs = replicationFactor
    op_tester.run(init_builder, reference, 'infer')
