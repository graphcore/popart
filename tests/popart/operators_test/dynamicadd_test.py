import itertools
import numpy as np
import popart
import torch
import pytest
import torch.nn.functional as F
from op_tester import op_tester


# Test dynamic addition, with overlapping regions, under training
# tensor -> A0 -> out0 -> A1 -> out1 -> A2 -> out2
#           ^             ^             ^
#           |             |             |
#           tensor0 * 1   tensor1 * 2   tensor2 * 3
# where tensor0, tensor1 and tensor2 are partially overlapping subregions
# of the out tensor, scaled with a different factor each.
def test_dynamicadd_training(op_tester):
    data = np.random.rand(4, 3, 5).astype(np.float32)
    data0 = np.random.rand(2, 3, 1).astype(np.float32)
    data1 = np.random.rand(2, 3, 1).astype(np.float32)
    data2 = np.random.rand(2, 3, 1).astype(np.float32)
    axes = [0, 2]
    sizes = [2, 1]

    def init_builder(builder):
        tensor = builder.addInitializedInputTensor(data)
        result = []
        add_tensors = [
            builder.addInitializedInputTensor(data0),
            builder.addInitializedInputTensor(data1),
            builder.addInitializedInputTensor(data2)
        ]
        out = tensor
        for i, slicex, slicey in [[0, 1, 2], [1, 2, 0], [2, 2, 2]]:
            index = builder.addInputTensor(
                np.asarray([slicex, slicey], np.uint32))
            add = builder.aiGraphcore.scale([add_tensors[i]], float(1 + i))
            out = builder.aiGraphcore.dynamicadd([out, index, add],
                                                 axes=axes,
                                                 sizes=sizes)
        result.append(out)

        sum = builder.aiOnnx.reducesum([out], axes=[0, 1, 2], keepdims=False)
        sum = builder.aiOnnx.unsqueeze([sum], axes=[0])

        builder.addOutputTensor(sum)
        result = [
            sum,
            popart.reservedGradientPrefix() + sum,
            popart.reservedGradientPrefix() + tensor,
            popart.reservedGradientPrefix() + add_tensors[0],
            popart.reservedGradientPrefix() + add_tensors[1],
            popart.reservedGradientPrefix() + add_tensors[2],
        ] + result
        return result

    def reference(ref_data):
        tensor = torch.tensor(data, requires_grad=True)
        tensor0 = torch.tensor(data0, requires_grad=True)
        tensor1 = torch.tensor(data1, requires_grad=True)
        tensor2 = torch.tensor(data2, requires_grad=True)

        outputs = []
        result = []

        t0p = torch.zeros(4, 3, 5)
        t0p[1:3, :, 2:3] = tensor0 * 1.0

        t1p = torch.zeros(4, 3, 5)
        t1p[2:4, :, 0:1] = tensor1 * 2.0

        t2p = torch.zeros(4, 3, 5)
        t2p[2:4, :, 2:3] = tensor2 * 3.0

        out = tensor + t0p + t1p + t2p
        outputs.append(out)
        result.append(out)

        sum = torch.unsqueeze(torch.sum(torch.stack(outputs)), dim=0)

        d__o = ref_data.getOutputTensorGrad(0)
        sum.backward(torch.tensor(d__o))

        result = [
            sum,
            torch.tensor(d__o), tensor.grad, tensor0.grad, tensor1.grad,
            tensor2.grad
        ] + result
        return result

    op_tester.passes = popart.PatternsLevel.ALL
    op_tester.run(init_builder, reference, 'train')
