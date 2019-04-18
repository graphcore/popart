import numpy as np
import pytest
import poponnx
import torch
from op_tester import op_tester


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

        loss = 0.1 * torch.norm(o, 1)
        loss.backward()
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

        # graph with gradient accumlation i.e. only update the weights after x passes
        for n in range(replicationFactor):
            o = module([a])
            loss = 0.1 * torch.norm(o, 1)
            loss.backward()

        # Update the weights
        optimizer.step()

        return [o, module.b.grad, module.B.data, module.c.grad, module.C.data]

    op_tester.passes = ['GemmDecomposition', 'PreUniRepl']
    op_tester.options.enableReplicatedGraphs = True
    op_tester.options.replicatedGraphCount = replicationFactor
    op_tester.device = "ipu_model"
    op_tester.numIPUs = replicationFactor
    op_tester.run(init_builder, reference, 'train')


def test_replication_infer(op_tester):

    A = np.ones((2, 4)).astype(np.float32)
    B = np.random.rand(4, 6).astype(np.float32)
    C = np.random.rand(2, 6).astype(np.float32)

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
        o = module([a])

        return [o]

    op_tester.passes = ['GemmDecomposition', 'PreUniRepl']
    op_tester.options.enableReplicatedGraphs = True
    op_tester.options.replicatedGraphCount = replicationFactor
    op_tester.device = "ipu_model"
    op_tester.numIPUs = replicationFactor
    op_tester.run(init_builder, reference, 'infer')
