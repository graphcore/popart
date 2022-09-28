# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import math
import numpy as np
import popart
import torch


def test_lrn(op_tester):
    alpha = 1e-4
    beta = 0.75
    bias = 1.0
    data = np.random.rand(2, 9, 4, 3, 2).astype(np.float32)
    nsizes = range(1, 12)

    def init_builder(builder):
        tensor = builder.addInputTensor(data)
        result = []
        for nsize in nsizes:
            out = builder.aiOnnx.lrn(
                [tensor],
                alpha=alpha,
                beta=beta,
                bias=bias,
                size=nsize,
                debugContext=f"test_lrn_{nsize}",
            )
            builder.addOutputTensor(out)
            result.append(out)
        return result

    def reference(_):  # ref_data is an unused argument
        result = []
        for nsize in nsizes:
            square_sum = np.zeros_like(data).astype(np.float32)
            for n, c, d0, d1, d2 in np.ndindex(data.shape):
                square_sum[n, c, d0, d1, d2] = sum(
                    data[
                        n,
                        max(0, c - int(math.floor((nsize - 1) / 2))) : min(
                            data.shape[1], c + int(math.ceil((nsize - 1) / 2)) + 1
                        ),
                        d0,
                        d1,
                        d2,
                    ]
                    ** 2
                )
            lrn_out_ref = data / ((bias + (alpha / nsize) * square_sum) ** beta)
            result.append(lrn_out_ref)
        return result

    op_tester.setPatterns(["PreUniRepl"], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, "infer")


def test_lrn_training(op_tester):
    op_tester.rtol = 1e-03
    op_tester.atol = 1e-08

    alpha = 1e-4
    beta = 0.75
    bias = 1.0
    data = np.random.rand(2, 9, 4, 3, 2).astype(np.float32)
    nsizes = range(1, 12)

    def init_builder(builder):
        result = []
        for nsize in nsizes:
            tensor = builder.addInputTensor(data)
            out = builder.aiOnnx.lrn(
                [tensor],
                alpha=alpha,
                beta=beta,
                bias=bias,
                size=nsize,
                debugContext=f"test_lrn_{nsize}",
            )
            result.append(out)
            result.append(popart.reservedGradientPrefix() + tensor)
        sum = builder.aiOnnx.sum(
            [
                builder.aiOnnx.reducesum(
                    [r],
                    axes=range(len(data.shape)),
                    keepdims=False,
                    debugContext="test_reducesum_all",
                )
                for r in result[0::2]
            ],
            debugContext="test_sum",
        )
        reshaped_sum = builder.aiOnnx.unsqueeze(
            [sum], axes=[0], debugContext="test_reshape"
        )
        builder.addOutputTensor(reshaped_sum)
        result = [reshaped_sum, popart.reservedGradientPrefix() + reshaped_sum] + result
        return result

    def reference(ref_data):
        result = []
        for nsize in nsizes:
            tensor = torch.tensor(data, requires_grad=True)
            out = torch.nn.LocalResponseNorm(nsize, alpha=alpha, beta=beta, k=bias)(
                tensor
            )
            result.append(out)
            result.append(tensor)

        sum = torch.unsqueeze(
            torch.sum(torch.stack([torch.sum(r) for r in result[0::2]])), dim=0
        )

        d__o = ref_data.getOutputTensorGrad(0)
        sum.backward(torch.tensor(d__o))
        result[1::2] = [r.grad for r in result[1::2]]

        result = [sum, sum.grad] + result
        return result

    op_tester.setPatterns(["OpToIdentity"], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, "train")
