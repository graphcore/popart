# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
import torch

# `import test_util` requires adding to sys.path
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu
# pylint is disabled as op_tester is used as a fixture
from operators_test.conftest import op_tester  # pylint: disable=unused-import


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

        return [
            o,
            popart.reservedGradientPrefix() + i2, i2, i3,
            "scaledLearningRate0___default___FLOAT",
            "weightDecayScaleFactor0___default___FLOAT",
            popart.reservedGradientPrefix() + o
        ]

    def reference(ref_data):
        class Module(torch.nn.Module):
            def __init__(self):
                super(Module, self).__init__()

                self.B = torch.nn.Parameter(torch.tensor(B),
                                            requires_grad=True)
                self.C = torch.nn.Parameter(torch.tensor(C),
                                            requires_grad=True)
                self.matmul = torch.matmul

            def forward(self, inputs):
                x = 1.0 * self.matmul(inputs[0], self.B) + 1.0 * self.C
                return x

        module = Module()

        a = torch.tensor(A, requires_grad=True)

        # train
        o = module([a])
        d__o = ref_data.getOutputTensorGrad(0)
        o.backward(torch.tensor(d__o))
        optimizer = torch.optim.SGD(module.parameters(), lr=0.01)
        optimizer.step()

        return [
            o, module.B.grad, module.B.data, module.C.data,
            np.float32(0.01),
            np.float32(1.0), None
        ]

    op_tester.numIPUs = 1
    op_tester.setPatterns([
        'MulArgGradOp', 'DecomposeBinaryConstScalar', 'PreUniRepl',
        'MatMulRhsGradOp'
    ],
                          enableRuntimeAsserts=False)
    op_tester.run(init_builder,
                  reference,
                  'train',
                  optimizer=popart.SGD({"defaultLearningRate": (0.01, False)}))


replicationFactor = 4


@tu.requires_ipu
def test_weight_update_replicated(op_tester):

    A = np.random.rand(2, 4).astype(np.float32)
    B = np.ones((4, 6)).astype(np.float32)
    C = np.random.rand(2, 6).astype(np.float32)

    alpha = np.random.random(1).astype(np.float32)[0]
    beta = np.random.random(1).astype(np.float32)[0]
    transA = False
    transB = False

    def init_builder(builder):
        i1 = builder.addInputTensor(A)
        i2 = builder.addInitializedInputTensor(B)
        i3 = builder.addInitializedInputTensor(C)
        o = builder.aiOnnx.gemm([i1, i2, i3], alpha, beta, transA, transB)
        builder.addOutputTensor(o)

        return [
            o,
            popart.reservedGradientPrefix() + i2, i2,
            popart.reservedGradientPrefix() + i3, i3,
            "scaledLearningRate0___default___FLOAT",
            "weightDecayScaleFactor0___default___FLOAT"
        ]

    def reference(_):  # ref_data is an unused argument
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

        optimizer = torch.optim.SGD(module.parameters(),
                                    lr=0.01,
                                    weight_decay=0.0,
                                    momentum=0.0)

        a = torch.tensor(A, requires_grad=True)

        optimizer.zero_grad()

        outputs = ()

        # graph with gradient accumlation i.e. only update the weights after x passes
        for n in range(replicationFactor):
            # adding n as offset, as op_tester expects
            o = module([a + n])
            outputs = outputs + (o, )
            loss = torch.nn.L1Loss(reduction="sum")
            target = torch.zeros(o.size())
            output = loss(o, target)
            output.backward()

        # Update the weights
        optimizer.step()

        # Add dimension to each output so we can concatenate them
        outputs = tuple(map(lambda x: torch.unsqueeze(x, 0), outputs))

        return [
            torch.cat(outputs), module.b.grad, module.B.data, module.c.grad,
            module.C.data,
            np.array([0.01, 0.01, 0.01, 0.01], np.float32),
            np.array([1, 1, 1, 1], np.float32)
        ]

    op_tester.lossReduction = popart.ReductionType.Sum
    op_tester.setPatterns([
        'DecomposeBinaryConstScalar', 'PreUniRepl', 'MatMulRhsGradOp',
        'MulArgGradOp'
    ],
                          enableRuntimeAsserts=False)
    op_tester.options.enableReplicatedGraphs = True
    op_tester.options.replicatedGraphCount = replicationFactor
    # Cant do opxModifyChecking wich replicated graphs.
    op_tester.options.opxModifyChecking = False
    op_tester.numIPUs = replicationFactor
    op_tester.run(init_builder,
                  reference,
                  'train',
                  optimizer=popart.SGD({"defaultLearningRate": (0.01, False)}))


@tu.requires_ipu
def test_replication_infer(op_tester):

    # 2 samples per device
    A = np.random.rand(2, 7).astype(np.float32)
    B = np.random.rand(7, 6).astype(np.float32)
    C = np.random.rand(1, 6).astype(np.float32)

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

        return [o]

    def reference(_):  # ref_data is an unused argument
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

        # forward
        # Run the pytorch module multiple times to simulate the same
        # behaviour as popart. The offsets (with corresponding offsets
        # in op_tester) ensure that the samples are distinct between replicas
        o1 = module([a + 0.])
        o2 = module([a + 1.])
        o3 = module([a + 2.])
        o4 = module([a + 3.])

        return [
            torch.cat((torch.unsqueeze(o1, 0), torch.unsqueeze(
                o2, 0), torch.unsqueeze(o3, 0), torch.unsqueeze(o4, 0)))
        ]

    op_tester.setPatterns(['DecomposeBinaryConstScalar', 'PreUniRepl'],
                          enableRuntimeAsserts=False)
    op_tester.options.enableReplicatedGraphs = True
    op_tester.options.replicatedGraphCount = replicationFactor
    op_tester.numIPUs = replicationFactor
    # Cant do opxModifyChecking wich replicated graphs.
    op_tester.options.opxModifyChecking = False
    op_tester.run(init_builder, reference, 'infer')


@tu.requires_ipu
def test_identity_loss_grad_replication():
    """
    Model:
                        (mean reduce)
    t0 -- Matmul - t2 - IdentityLoss - t3
    t1 ----'
    """
    t_data = np.random.rand(8).astype(np.float32)

    def getIdentityLossGradTensor(opToIdentityPattern):
        builder = popart.Builder()
        t0 = builder.addInputTensor("FLOAT", [1, 4])
        # t1_data = np.random.rand(4, 1).astype(np.float32)
        t1 = builder.addInputTensor("FLOAT", [4, 1])

        t2 = builder.aiOnnx.matmul([t0, t1])
        t3 = builder.aiGraphcore.identityloss(
            [t2], reduction=popart.ReductionType.Mean)

        opts = popart.SessionOptions()
        opts.enableReplicatedGraphs = True
        opts.replicatedGraphCount = 2

        patterns = popart.Patterns()
        patterns.OpToIdentity = opToIdentityPattern

        with tu.create_test_device(numIpus=2) as device:
            session = popart.TrainingSession(
                fnModel=builder.getModelProto(),
                deviceInfo=device,
                dataFlow=popart.DataFlow(
                    1, [t3, popart.reservedGradientPrefix() + t0]),
                loss=t3,
                optimizer=popart.ConstSGD(0.1),
                userOptions=opts,
                patterns=patterns)

            anchors = session.initAnchorArrays()
            session.prepareDevice()
            stepio = popart.PyStepIO({t0: t_data, t1: t_data}, anchors)
            session.run(stepio)
        return anchors[popart.reservedGradientPrefix() + t0]

    t2_grad = getIdentityLossGradTensor(True)
    t2_grad_no_op_to_id = getIdentityLossGradTensor(False)
    assert np.array_equal(t2_grad, t2_grad_no_op_to_id)
