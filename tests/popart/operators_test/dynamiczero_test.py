import itertools
import numpy as np
import popart
import torch
import pytest
import torch.nn.functional as F
from op_tester import op_tester


# Test training with zeroing out a dynamic region of a tensor, e.g.
# tensor[:, 4:8, 2:3, :] = 0.0
# tensor += inject_tensor
# tensor[:, 8:12, 0:1, :] = 0.0
# where the regions are selected by a tensor offset.
# A weight inject tensor is added in between zeroing out regions, to test
# if the gradients of inject_tensor are only affected by the second zeroing,
# and the gradients of the tensor are affected by both zeroing operations.
def test_dynamiczero_training(op_tester):
    data = np.random.rand(2, 12, 3, 3).astype(np.float32)
    inject_data = np.random.rand(2, 12, 3, 3).astype(np.float32)
    axes = [1, 2]
    sizes = [4, 1]

    def init_builder(builder):
        tensor = builder.addInitializedInputTensor(data)
        inject_tensor = builder.addInitializedInputTensor(inject_data)
        result = []
        out = tensor
        for i, slicex, slicey in [[0, 1, 2], [1, 2, 0]]:
            index = builder.addInputTensor(
                np.asarray([slicex * sizes[0], slicey * sizes[1]], np.uint32))
            out = builder.aiGraphcore.dynamiczero([out, index],
                                                  axes=axes,
                                                  sizes=sizes)
            if i == 0:
                out = builder.aiOnnx.add([out, inject_tensor])
        result.append(out)

        sum = builder.aiOnnx.reducesum([out],
                                       axes=[0, 1, 2, 3],
                                       keepdims=False)
        sum = builder.aiOnnx.unsqueeze([sum], axes=[0])

        builder.addOutputTensor(sum)
        result = [
            sum,
            popart.reservedGradientPrefix() + sum,
            popart.reservedGradientPrefix() + tensor,
            popart.reservedGradientPrefix() + inject_tensor,
        ] + result
        return result

    def reference(ref_data):
        tensor = torch.tensor(data, requires_grad=True)
        inject_tensor = torch.tensor(inject_data, requires_grad=True)
        outputs = []
        result = []
        mask0data = np.ones(data.shape)
        mask0data[:, 4:8, 2:3, :] = 0.0
        mask1data = np.ones(data.shape)
        mask1data[:, 8:12, 0:1, :] = 0.0
        mask0 = torch.tensor(mask0data)
        mask1 = torch.tensor(mask1data)
        out = (((tensor * mask0) + inject_tensor) * mask1)
        outputs.append(out)
        result.append(out)

        sum = torch.unsqueeze(torch.sum(torch.stack(outputs)), dim=0)

        d__o = ref_data.getOutputTensorGrad(0)
        sum.backward(torch.tensor(d__o))

        result = [sum,
                  torch.tensor(d__o), tensor.grad, inject_tensor.grad] + result
        return result

    op_tester.passes = popart.PatternsLevel.ALL
    op_tester.run(init_builder, reference, 'train')
