# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import numpy as np
import torch
import popart


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


def test_gemm_no_C(op_tester):
    A = np.random.rand(2, 4).astype(np.float32)
    B = np.random.rand(4, 6).astype(np.float32)
    _test_gemm(op_tester, A, B, None, 1.0, 1.0, False, False, onnx_version=11)


def test_gemm_grad_no_C(op_tester):
    A = np.random.rand(2, 4).astype(np.float32)
    B = np.random.rand(4, 6).astype(np.float32)
    _test_gemm_grad(op_tester,
                    A,
                    B,
                    None,
                    1.0,
                    1.0,
                    False,
                    False,
                    onnx_version=11)


def test_gemm_halfs(op_tester):
    A = np.random.rand(2, 4).astype(np.float16)
    B = np.random.rand(4, 6).astype(np.float16)
    C = np.random.rand(2, 6).astype(np.float16)
    _test_gemm(op_tester, A, B, C, 1.0, 1.0, False, False)


def test_gemm_transpose_ab_halfs(op_tester):
    A = np.random.rand(4, 2).astype(np.float16)
    B = np.random.rand(6, 4).astype(np.float16)
    C = np.random.rand(2, 6).astype(np.float16)
    _test_gemm(op_tester, A, B, C, 1.0, 1.0, True, True)


def _test_gemm(op_tester,
               A,
               B,
               C,
               alpha,
               beta,
               transA,
               transB,
               onnx_version=None):
    def init_builder(builder):
        i1 = builder.addInputTensor(A)
        i2 = builder.addInputTensor(B)
        if C is not None:
            i3 = builder.addInputTensor(C)
            o = builder.aiOnnx.gemm([i1, i2, i3], alpha, beta, transA, transB)
        else:
            o = builder.aiOnnx.gemm([i1, i2], alpha, beta, transA, transB)
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

        o = alpha * np.dot(a, b)
        if C is not None:
            o += beta * c
        return [o]

    op_tester.setPatterns(['DecomposeBinaryConstScalar'],
                          enableRuntimeAsserts=False)

    opsets = None
    if onnx_version is not None:
        opsets = {'ai.onnx': onnx_version, 'ai.graphcore': 1}
    # For a range around 1 the rounding error for halfs is approx eps = 0.0001.
    # Example 1/3 - (half(1/3) ~ 0.3333) = 0.0001.
    # Matmul A(n, k) B(k, m), element x is sum a_ik * b_kj.
    # Multiplication error a * b ~ 2*a*eps.
    # Expecting sumation error as random walk: 2*Sqrt(k)*eps ~ 0.002.
    if A.dtype == np.float16:
        op_tester.atol = 1e-2
    op_tester.run(init_builder, reference, 'infer', opsets=opsets)


def _test_gemm_grad(op_tester,
                    A,
                    B,
                    C,
                    alpha,
                    beta,
                    transA,
                    transB,
                    onnx_version=10):
    alpha = float(alpha)
    beta = float(beta)

    def init_builder(builder):
        i1 = builder.addInputTensor(A)
        i2 = builder.addInputTensor(B)
        if C is not None:
            i3 = builder.addInputTensor(C)
            o = builder.aiOnnx.gemm([i1, i2, i3], alpha, beta, transA, transB)
        else:
            o = builder.aiOnnx.gemm([i1, i2], alpha, beta, transA, transB)
        builder.addOutputTensor(o)

        result = [
            o,
            popart.reservedGradientPrefix() + i1,
            popart.reservedGradientPrefix() + i2,
        ]
        if C is not None:
            result.append(popart.reservedGradientPrefix() + i3)
        result.append(popart.reservedGradientPrefix() + o)
        return result

    def reference(ref_data):
        a = torch.tensor(A, requires_grad=True)
        b = torch.tensor(B, requires_grad=True)

        if transA:
            a = a.permute(1, 0)
        if transB:
            b = b.permute(1, 0)

        o = alpha * torch.matmul(a, b)
        if C is not None:
            c = torch.tensor(C, requires_grad=True)
            o += beta * c

        d__o = ref_data.getOutputTensorGrad(0)
        o.backward(torch.tensor(d__o))

        result = [o, a.grad, b.grad]
        if C is not None:
            result.append(c.grad)
        result.append(None)
        return result
        # return [o, a.grad, b.grad, c.grad, None]

    op_tester.setPatterns([
        'DecomposeBinaryConstScalar', 'PreUniRepl', 'MatMulLhsGradOp',
        'MatMulRhsGradOp', 'MulArgGradOp'
    ],
                          enableRuntimeAsserts=False)

    opsets = None
    if onnx_version is not None:
        opsets = {'ai.onnx': onnx_version, 'ai.graphcore': 1}
    if A.dtype == np.float16:
        op_tester.atol = 1e-2
    op_tester.run(init_builder, reference, 'train', opsets=opsets)
