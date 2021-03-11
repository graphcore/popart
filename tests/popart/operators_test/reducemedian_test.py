import itertools
import numpy as np
import popart
import pytest
import torch

from op_tester import op_tester

USE_DEFAULT_AXES = None

types_list = [np.float32, np.int32]
keepdims_list = [False, True]

test_reducemedian_values_axes = [USE_DEFAULT_AXES]
for i in range(5):
    test_reducemedian_values_axes += list(itertools.combinations(range(4), i))


@pytest.mark.parametrize("type", types_list)
@pytest.mark.parametrize("keepdims", keepdims_list)
@pytest.mark.parametrize("axes", test_reducemedian_values_axes)
def test_reducemedian_values(op_tester, type, keepdims, axes):
    # Reduction compared to numpy.median, note that numpy averages the two
    # medians in the case of even number of input values, so we choose the input
    # shape that ensures the odd number of values over which the reduction is
    # performed.
    if np.issubdtype(type, np.integer):
        data = np.random.randint(100, size=[1, 3, 5, 7], dtype=type)
    elif np.issubdtype(type, np.floating):
        data = np.random.randn(1, 3, 5, 7).astype(type)
    else:
        raise ValueError("Illegal input type")

    def init_builder(builder):
        tensor = builder.addInputTensor(data)
        out = builder.aiGraphcore.reducemedian(
            [tensor],
            axes=axes,
            keepdims=keepdims,
            debugContext='test_reducemedian_values_{}_{}'.format(
                axes, keepdims),
        )
        builder.addOutputTensor(out[0])
        return [out[0]]

    def reference(ref_data):
        return [np.median(data, axis=axes, keepdims=keepdims).astype(type)]

    op_tester.run(init_builder, reference, 'infer')


test_reducemedian_indices_1_axes = [[0], [1], [2], [3], [4]]


@pytest.mark.parametrize("type", types_list)
@pytest.mark.parametrize("keepdims", keepdims_list)
@pytest.mark.parametrize("axes", test_reducemedian_indices_1_axes)
def test_reducemedian_indices_1(op_tester, type, keepdims, axes):
    # Reduction over a single axis (compatible with torch).
    if np.issubdtype(type, np.integer):
        # Torch/PopART may not return the first occurrence of the median value
        # unless it is unique and the exact behaviour may be differ between the
        # two implementations.
        data = np.random.default_rng().choice(10000,
                                              size=[1, 3, 4, 7, 8],
                                              replace=False).astype(type)
    elif np.issubdtype(type, np.floating):
        data = np.random.randn(1, 3, 4, 7, 8).astype(type)
    else:
        raise ValueError("Illegal input type")

    def init_builder(builder):
        tensor = builder.addInputTensor(data)
        out = builder.aiGraphcore.reducemedian(
            [tensor],
            axes=axes,
            keepdims=keepdims,
            debugContext='test_reducemedian_indices_1_{}_{}'.format(
                axes, keepdims),
        )
        builder.addOutputTensor(out[1])
        return [out[1]]

    def reference(ref_data):
        tensor = torch.tensor(data)
        out = torch.median(tensor, dim=axes[0], keepdim=keepdims)
        return [out.indices.numpy().astype(np.int32)]

    op_tester.run(init_builder, reference, 'infer')


test_reducemedian_indices_2_axes = [[0, 1], USE_DEFAULT_AXES]


@pytest.mark.parametrize("type", types_list)
@pytest.mark.parametrize("keepdims", keepdims_list)
@pytest.mark.parametrize("axes", test_reducemedian_indices_2_axes)
def test_reducemedian_indices_2(op_tester, type, keepdims, axes):
    # Reduction over multiple axes (incompatible with torch).
    data = np.asarray([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]],
                      dtype=type)

    def init_builder(builder):
        tensor = builder.addInputTensor(data)
        out = builder.aiGraphcore.reducemedian(
            [tensor],
            axes=axes,
            keepdims=keepdims,
            debugContext='test_reducemedian_indices_2_{}_{}'.format(
                axes, keepdims),
        )
        builder.addOutputTensor(out[1])
        return [out[1]]

    def reference(ref_data):
        tensor = np.asarray([7], dtype=np.int32).reshape(())
        if keepdims:
            out = tensor.reshape((1, 1))
        else:
            out = tensor.copy()
        return [out]

    op_tester.run(init_builder, reference, 'infer')


test_reducemedian_training_1_axes = [[0], [1], [2], [3]]


@pytest.mark.parametrize("type", types_list)
@pytest.mark.parametrize("keepdims", keepdims_list)
@pytest.mark.parametrize("axes", test_reducemedian_training_1_axes)
def test_reducemedian_training_1(op_tester, type, keepdims, axes):
    # Reduction over a single axis (compatible with torch).
    if np.issubdtype(type, np.integer):
        # Torch/PopART may not return the first occurrence of the median value
        # unless it is unique and the exact behaviour may be differ between the
        # two implementations.
        data = np.random.default_rng().choice(10000,
                                              size=[3, 4, 7, 8],
                                              replace=False).astype(type)
    elif np.issubdtype(type, np.floating):
        data = np.random.randn(3, 4, 7, 8).astype(type)
    else:
        raise ValueError("Illegal input type")

    def init_builder(builder):
        tensor = builder.addInputTensor(data)
        out = builder.aiGraphcore.reducemedian(
            [tensor],
            axes=axes,
            keepdims=keepdims,
            debugContext='test_reducemedian_training_1_{}_{}'.format(
                axes, keepdims),
        )
        sum = builder.aiOnnx.reducesum(
            [out[0]],
            keepdims=False,
            debugContext='test_reducemedian_training_1_{}_{}'.format(
                axes, keepdims),
        )
        builder.addOutputTensor(sum)
        return [sum, out[0], popart.reservedGradientPrefix() + tensor]

    def reference(ref_data):
        tensor = torch.tensor(data)
        # Torch does not support gradients of integer tensors but in the context
        # of median, where the gradient from the top is simply scattered along
        # median indices it is a valid operation.
        tensor_float = tensor.type(torch.float32)
        tensor_float.requires_grad = True
        out = torch.median(tensor_float, dim=axes[0], keepdim=keepdims)
        sum = out.values.sum()
        sum.backward()
        return [
            sum.type(tensor.dtype),
            out.values.type(tensor.dtype),
            tensor_float.grad.type(tensor.dtype)
        ]

    op_tester.run(init_builder, reference, 'train')


test_reducemedian_training_2_axes = [[0, 1], USE_DEFAULT_AXES]


@pytest.mark.parametrize("type", types_list)
@pytest.mark.parametrize("keepdims", keepdims_list)
@pytest.mark.parametrize("axes", test_reducemedian_training_2_axes)
def test_reducemedian_training_2(op_tester, type, keepdims, axes):
    # Reduction over multiple axes (incompatible with torch).
    data = np.asarray([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]],
                      dtype=type)

    def init_builder(builder):
        tensor = builder.addInputTensor(data)
        out = builder.aiGraphcore.reducemedian(
            [tensor],
            axes=axes,
            keepdims=keepdims,
            debugContext='test_reducemedian_training_2_{}_{}'.format(
                axes, keepdims),
        )
        sum = builder.aiOnnx.reducesum(
            [out[0]],
            keepdims=False,
            debugContext='test_reducemedian_training_2_{}_{}'.format(
                axes, keepdims),
        )
        builder.addOutputTensor(sum)
        return [sum, out[0], popart.reservedGradientPrefix() + tensor]

    def reference(ref_data):
        tensor = np.asarray([7], dtype=type).reshape(())
        grad = np.zeros(data.shape, dtype=type)
        grad[1, 2] = 1
        if keepdims:
            out = tensor.reshape((1, 1))
        else:
            out = tensor.copy()
        return [tensor, out, grad]

    op_tester.run(init_builder, reference, 'train')
