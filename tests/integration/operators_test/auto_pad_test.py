# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

# This test is disabled as auto_pad is not exposed in the builder.
# However if you wish to test the auto_pad feature, add the options
# to the builder and this test should pass. There are also backend
# tests that test auto_pad via onnx models
import itertools
from typing import Sequence, Text

import numpy as np  # type: ignore
import pytest
import torch

import popart
from op_tester import op_tester


# Taken from onnx/onnx/backend/test/case/node/pool_op_common.py
def get_pad_shape(
        auto_pad,  # type: Text
        input_spatial_shape,  # type: Sequence[int]
        kernel_spatial_shape,  # type: Sequence[int]
        strides_spatial,  # type: Sequence[int]
        output_spatial_shape  # type: Sequence[int]
):  # type: (...) -> Sequence[int]
    pad_shape = [0] * len(input_spatial_shape)
    if auto_pad in ('SAME_UPPER', 'SAME_LOWER'):
        for i in range(len(input_spatial_shape)):
            pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial[i] + \
                kernel_spatial_shape[i] - input_spatial_shape[i]
    elif auto_pad == 'VALID':
        pass
    return pad_shape


def get_output_shape(
        auto_pad,  # type: Text
        input_spatial_shape,  # type: Sequence[int]
        kernel_spatial_shape,  # type: Sequence[int]
        strides_spatial  # type: Sequence[int]
):  # type: (...) -> Sequence[int]
    out_shape = [0] * len(input_spatial_shape)
    if auto_pad in ('SAME_UPPER', 'SAME_LOWER'):
        for i in range(len(input_spatial_shape)):
            out_shape[i] = int(
                np.ceil(
                    float(input_spatial_shape[i]) / float(strides_spatial[i])))
    elif auto_pad == 'VALID':
        for i in range(len(input_spatial_shape)):
            out_shape[i] = int(
                np.ceil(
                    float(input_spatial_shape[i] -
                          (kernel_spatial_shape[i] - 1)) /
                    float(strides_spatial[i])))
    return out_shape


def pool(
        padded,  # type: np.ndarray
        x_shape,  # type: Sequence[int]
        kernel_shape,  # type: Sequence[int]
        strides_shape,  # type: Sequence[int]
        out_shape,  # type: Sequence[int]
        pad_shape,  # type: Sequence[int]
        pooling_type,  # type: Text
        count_include_pad=0  # type: int
):  # type: (...) -> np.ndarray
    spatial_size = len(x_shape) - 2
    y = np.zeros([x_shape[0], x_shape[1]] + list(out_shape))

    for shape in itertools.product(
            range(x_shape[0]), range(x_shape[1]), *[
                range(
                    int((x_shape[i + 2] + pad_shape[i] - kernel_shape[i]) /
                        strides_shape[i] + 1)) for i in range(spatial_size)
            ]):
        window = padded[shape[0], shape[1]]
        window_vals = np.array([
            window[i] for i in list(
                itertools.product(*[
                    range(strides_shape[i] *
                          shape[i + 2], strides_shape[i] * shape[i + 2] +
                          kernel_shape[i]) for i in range(spatial_size)
                ]))
        ])
        if pooling_type == 'AVG':
            f = np.average
        elif pooling_type == 'MAX':
            f = np.max
        else:
            raise NotImplementedError(
                'Pooling type {} does not support. Should be AVG, MAX'.format(
                    pooling_type))

        if count_include_pad == 1 and pooling_type == 'AVG':
            y[shape] = f(window_vals)
        else:
            y[shape] = f(window_vals[np.where(~np.isnan(window_vals))])
    return y.astype(np.float32)


def test_average_pool_1_auto_pad(op_tester):
    x = np.random.randn(1, 3, 32).astype(np.float32)
    x_shape = np.shape(x)
    kernel_shape = [2]
    strides = [1]
    out_shape = get_output_shape('VALID', x_shape[2:], kernel_shape, strides)
    padded = x

    def init_builder(builder):
        i1 = builder.addInputTensor(x)
        o = builder.aiOnnx.averagepool([i1],
                                       kernel_shape=kernel_shape,
                                       auto_pad="VALID",
                                       count_include_pad=0,
                                       strides=strides)
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        out = pool(padded, x_shape, kernel_shape, strides, out_shape, [0],
                   'AVG')

        return [out]

    op_tester.run(init_builder, reference)


def test_average_pool_2_auto_pad(op_tester):
    x = np.random.randn(1, 3, 32, 32).astype(np.float32)
    x_shape = np.shape(x)
    kernel_shape = (2, 2)
    strides = (1, 1)
    out_shape = get_output_shape('VALID', x_shape[2:], kernel_shape, strides)
    padded = x

    def init_builder(builder):
        i1 = builder.addInputTensor(x)
        o = builder.aiOnnx.averagepool([i1],
                                       kernel_shape=kernel_shape,
                                       auto_pad="VALID",
                                       count_include_pad=0,
                                       strides=strides)
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        out = pool(padded, x_shape, kernel_shape, strides, out_shape, [0, 0],
                   'AVG')

        return [out]

    op_tester.run(init_builder, reference)


def test_average_pool_3_auto_pad(op_tester):
    """
        input_shape: [1, 3, 32, 32]
        output_shape: [1, 3, 32, 32]
        pad_shape: [1, 1] -> [0, 1, 0, 1] by axis
    """
    x = np.random.randn(1, 3, 32, 32).astype(np.float32)
    x_shape = np.shape(x)
    kernel_shape = (2, 2)
    strides = (1, 1)

    def init_builder(builder):
        i1 = builder.addInputTensor(x)
        o = builder.aiOnnx.averagepool([i1],
                                       kernel_shape=kernel_shape,
                                       auto_pad="SAME_UPPER",
                                       strides=strides)
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        out_shape = get_output_shape('SAME_UPPER', x_shape[2:], kernel_shape,
                                     strides)
        pad_shape = get_pad_shape('SAME_UPPER', x_shape[2:], kernel_shape,
                                  strides, out_shape)
        pad_top = pad_shape[0] // 2
        pad_bottom = pad_shape[0] - pad_top
        pad_left = pad_shape[1] // 2
        pad_right = pad_shape[1] - pad_left
        padded = np.pad(
            x, ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
            mode='constant',
            constant_values=np.nan)

        out = pool(padded, x_shape, kernel_shape, strides, out_shape,
                   pad_shape, 'AVG')

        return [out]

    op_tester.run(init_builder, reference)


def test_average_pool_4_auto_pad(op_tester):
    """
        input_shape: [1, 3, 32, 32]
        output_shape: [1, 3, 32, 32]
        pad_shape: [1, 1] -> [0, 1, 0, 1] by axis
    """
    x = np.random.randn(1, 3, 32, 32).astype(np.float32)
    x_shape = np.shape(x)
    kernel_shape = (2, 2)
    strides = (1, 1)

    def init_builder(builder):
        i1 = builder.addInputTensor(x)
        o = builder.aiOnnx.averagepool([i1],
                                       kernel_shape=kernel_shape,
                                       auto_pad="SAME_LOWER",
                                       strides=strides)
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        out_shape = get_output_shape('SAME_LOWER', x_shape[2:], kernel_shape,
                                     strides)
        pad_shape = get_pad_shape('SAME_LOWER', x_shape[2:], kernel_shape,
                                  strides, out_shape)
        pad_bottom = pad_shape[0] // 2
        pad_top = pad_shape[0] - pad_bottom
        pad_right = pad_shape[1] // 2
        pad_left = pad_shape[1] - pad_right
        padded = np.pad(
            x, ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
            mode='constant',
            constant_values=np.nan)

        out = pool(padded, x_shape, kernel_shape, strides, out_shape,
                   pad_shape, 'AVG')

        return [out]

    op_tester.atol = 1e-5
    op_tester.rtol = 1e-8
    op_tester.run(init_builder, reference)


def test_max_pool_1_auto_pad(op_tester):
    """
    input_shape: [1, 1, 5, 5]
    output_shape: [1, 1, 3, 3]
    pad_shape: [2, 2] -> [1, 1, 1, 1] by axis
    """
    x = np.array([[[
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20],
        [21, 22, 23, 24, 25],
    ]]]).astype(np.float32)
    kernel_shape = (3, 3)
    strides = (2, 2)

    def init_builder(builder):
        i1 = builder.addInputTensor(x)
        o = builder.aiOnnx.maxpool([i1],
                                   num_outputs=1,
                                   kernel_shape=kernel_shape,
                                   auto_pad="SAME_UPPER",
                                   strides=strides)[0]
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        out = np.array([[[[7, 9, 10], [17, 19, 20], [22, 24,
                                                     25]]]]).astype(np.float32)

        return [out]

    op_tester.run(init_builder, reference)


def test_max_pool_2_auto_pad(op_tester):
    """
        input_shape: [1, 3, 32, 32]
        output_shape: [1, 3, 32, 32]
        pad_shape: [1, 1] -> [0, 1, 0, 1] by axis
    """
    x = np.random.randn(1, 3, 32, 32).astype(np.float32)
    x_shape = np.shape(x)
    kernel_shape = (2, 2)
    strides = (1, 1)

    def init_builder(builder):
        i1 = builder.addInputTensor(x)
        o = builder.aiOnnx.maxpool([i1],
                                   num_outputs=1,
                                   kernel_shape=kernel_shape,
                                   auto_pad="SAME_UPPER",
                                   strides=strides)[0]
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        out_shape = get_output_shape('SAME_UPPER', x_shape[2:], kernel_shape,
                                     strides)
        pad_shape = get_pad_shape('SAME_UPPER', x_shape[2:], kernel_shape,
                                  strides, out_shape)
        pad_top = pad_shape[0] // 2
        pad_bottom = pad_shape[0] - pad_top
        pad_left = pad_shape[1] // 2
        pad_right = pad_shape[1] - pad_left
        padded = np.pad(
            x, ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
            mode='constant',
            constant_values=np.nan)

        out = pool(padded, x_shape, kernel_shape, strides, out_shape,
                   pad_shape, 'MAX')

        return [out]

    op_tester.run(init_builder, reference)
