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
from operators_test.op_tester import op_tester


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
            "weightDecayScaleFactor0___default___FLOAT"
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

        # forward
        o = module([a])

        return [
            o, module.B.grad, module.B.data, module.C.data,
            np.float32(0.01),
            np.float32(1.0)
        ]

    op_tester.device = tu.create_test_device()
    op_tester.numIPUs = 1
    op_tester.setPatterns(
        ['GemmDecomposition', 'PreUniRepl', 'MatMulRhsGradOp', 'OpToReshape'],
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
    op_tester.setPatterns(
        ['GemmDecomposition', 'PreUniRepl', 'MatMulRhsGradOp', 'OpToReshape'],
        enableRuntimeAsserts=False)
    op_tester.options.enableReplicatedGraphs = True
    op_tester.options.replicatedGraphCount = replicationFactor
    op_tester.device = tu.create_test_device(numIpus=replicationFactor)
    if not op_tester.device:
        raise RuntimeError(
            "Failed to acquire IPU device in training graph replication test")

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

    op_tester.setPatterns(['GemmDecomposition', 'PreUniRepl', 'OpToReshape'],
                          enableRuntimeAsserts=False)
    op_tester.options.enableReplicatedGraphs = True
    op_tester.options.replicatedGraphCount = replicationFactor
    op_tester.device = tu.create_test_device(replicationFactor)
    if not op_tester.device:
        raise RuntimeError(
            "Failed to acquire IPU device in inference graph replication test")

    op_tester.numIPUs = replicationFactor
    op_tester.run(init_builder, reference, 'infer')
