import numpy as np
import torch
from op_tester import op_tester


def test_gemm_basic(op_tester):
    A = np.random.rand(2, 4).astype(np.float32)
    B = np.random.rand(4, 6).astype(np.float32)
    C = np.random.rand(2, 6).astype(np.float32)
    _test_gemm(op_tester, A, B, C, 1.0, 1.0, False, False)


def test_gemm_scale(op_tester):
    A = np.random.rand(2, 4).astype(np.float32)
    B = np.random.rand(4, 6).astype(np.float32)
    C = np.random.rand(2, 6).astype(np.float32)
    alpha = np.random.random(1).astype(np.float32)[0]
    beta = np.random.random(1).astype(np.float32)[0]
    _test_gemm(op_tester, A, B, C, alpha, beta, False, False)


def test_gemm_transpose_a(op_tester):
    A = np.random.rand(4, 2).astype(np.float32)
    B = np.random.rand(4, 6).astype(np.float32)
    C = np.random.rand(2, 6).astype(np.float32)
    _test_gemm(op_tester, A, B, C, 1.0, 1.0, True, False)


def test_gemm_transpose_b(op_tester):
    A = np.random.rand(2, 4).astype(np.float32)
    B = np.random.rand(6, 4).astype(np.float32)
    C = np.random.rand(2, 6).astype(np.float32)
    _test_gemm(op_tester, A, B, C, 1.0, 1.0, False, True)


def test_gemm_transpose_ab(op_tester):
    A = np.random.rand(4, 2).astype(np.float32)
    B = np.random.rand(6, 4).astype(np.float32)
    C = np.random.rand(2, 6).astype(np.float32)
    _test_gemm(op_tester, A, B, C, 1.0, 1.0, True, True)


def test_gemm_basic_grad(op_tester):
    A = np.random.rand(2, 4).astype(np.float32)
    B = np.random.rand(4, 6).astype(np.float32)
    C = np.random.rand(2, 6).astype(np.float32)
    _test_gemm_grad(op_tester, A, B, C, 1.0, 1.0, False, False)


def test_gemm_grad(op_tester):
    A = np.random.rand(4, 2).astype(np.float32)
    B = np.random.rand(6, 4).astype(np.float32)
    C = np.random.rand(2, 6).astype(np.float32)
    _test_gemm_grad(op_tester, A, B, C, 1.0, 1.0, True, True)


def test_gemm_grad_scale(op_tester):
    A = np.random.rand(2, 4).astype(np.float32)
    B = np.random.rand(4, 6).astype(np.float32)
    C = np.random.rand(2, 6).astype(np.float32)
    alpha = np.random.random(1).astype(np.float32)[0]
    beta = np.random.random(1).astype(np.float32)[0]
    _test_gemm_grad(op_tester, A, B, C, alpha, beta, False, False)


def _test_gemm(op_tester, A, B, C, alpha, beta, transA, transB):
    def init_builder(builder):
        i1 = builder.addInputTensor(A)
        i2 = builder.addInputTensor(B)
        i3 = builder.addInputTensor(C)
        o = builder.gemm([i1, i2, i3], alpha, beta, transA, transB)
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        a = A
        b = B
        c = C

        if transA:
            a = np.transpose(a)
        if transB:
            b = np.transpose(b)

        o = alpha * np.dot(a, b) + beta * c
        return [o]

    op_tester.passes = ['GemmDecomposition']
    op_tester.run(init_builder, reference, 'infer')


def _test_gemm_grad(op_tester, A, B, C, alpha, beta, transA, transB):
    alpha = float(alpha)
    beta = float(beta)

    def init_builder(builder):
        i1 = builder.addInputTensor(A)
        i2 = builder.addInputTensor(B)
        i3 = builder.addInputTensor(C)
        o = builder.gemm([i1, i2, i3], alpha, beta, transA, transB)
        builder.addOutputTensor(o)
        return [o, 'd__' + i1, 'd__' + i2, 'd__' + i3, 'd__' + o]

    def reference(ref_data):
        a = torch.tensor(A, requires_grad=True)
        b = torch.tensor(B, requires_grad=True)
        c = torch.tensor(C, requires_grad=True)

        if transA:
            a = a.permute(1, 0)
        if transB:
            b = b.permute(1, 0)

        o = alpha * torch.matmul(a, b) + beta * c
        d__o = ref_data.getOutputTensorGrad(0)
        o.backward(torch.tensor(d__o))
        return [o, a.grad, b.grad, c.grad, None]

    op_tester.passes = ['GemmDecomposition', 'PreUniRepl']
    op_tester.run(init_builder, reference, 'train')
