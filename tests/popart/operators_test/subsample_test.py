# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import numpy as np
import pytest
import popart
from op_tester import op_tester


def _subsample_helper(op_tester, input, strides, output, grad_ouput):
    # create test data
    d1 = input

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiGraphcore.subsample([i1], strides)
        builder.addOutputTensor(o)
        return [o, popart.reservedGradientPrefix() + i1]

    def reference(ref_data):
        return [output, grad_ouput]

    op_tester.patterns = ['PreUniRepl', 'SqrtGradOp']
    op_tester.run(init_builder, reference, 'train')


def test_subsample1(op_tester):
    _subsample_helper(
        op_tester,
        np.array([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]],
                 dtype=np.float32), [2, 2],
        np.array([[1, 3], [3, 5]], dtype=np.float32),
        np.array(
            [[0.1, 0, 0.1, 0], [0, 0, 0, 0], [0.1, 0, 0.1, 0], [0, 0, 0, 0]],
            dtype=np.float32))


def test_subsample2(op_tester):
    _subsample_helper(
        op_tester,
        np.array([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]],
                 dtype=np.float32), [1, 1],
        np.array([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]],
                 dtype=np.float32),
        np.array([[0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1],
                  [0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]],
                 dtype=np.float32))


def test_subsample3(op_tester):
    _subsample_helper(
        op_tester,
        np.array([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]],
                 dtype=np.float32), [2, 1],
        np.array([
            [1, 2, 3, 4],
            [3, 4, 5, 6],
        ], dtype=np.float32),
        np.array([[0.1, 0.1, 0.1, 0.1], [0, 0, 0, 0], [0.1, 0.1, 0.1, 0.1],
                  [0, 0, 0, 0]],
                 dtype=np.float32))


def test_subsample4(op_tester):
    _subsample_helper(
        op_tester,
        np.array([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]],
                 dtype=np.float32), [3, 3],
        np.array([[1, 4], [4, 7]], dtype=np.float32),
        np.array(
            [[0.1, 0, 0, 0.1], [0, 0, 0, 0], [0, 0, 0, 0], [0.1, 0, 0, 0.1]],
            dtype=np.float32))


def test_subsample5(op_tester):
    _subsample_helper(
        op_tester,
        np.array([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]],
                 dtype=np.float32), [4, 4], np.array([[1]], dtype=np.float32),
        np.array([[0.1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                 dtype=np.float32))


# Test the error case where there is a 0 stride
def test_subsample6(op_tester):

    with pytest.raises(popart.popart_exception) as e_info:

        _subsample_helper(
            op_tester,
            np.array([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]],
                     dtype=np.float32), [4, 0],
            np.array([[1]], dtype=np.float32),
            np.array(
                [[0.1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                dtype=np.float32))

    assert (e_info.value.args[0].startswith(
        "Strides invalid. 0 stride at index 1"))
