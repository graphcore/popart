# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
import torch
import pytest
from op_tester import op_tester

# `import test_util` requires adding to sys.path
import sys
from pathlib import Path
sys.path.append(Path(__file__).resolve().parent.parent)
import test_util as tu


def test_matmul_grouped_1(op_tester):
    d1 = np.random.rand(2, 1, 4, 5, 1, 7, 8).astype(np.float32)
    d2 = np.random.rand(2, 3, 1, 5, 6, 8, 9).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        o = builder.aiOnnx.matmul([i1, i2])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        out = np.matmul(d1, d2)
        return [out]

    op_tester.run(init_builder, reference)


def test_matmul_grouped_2(op_tester):
    d1 = np.random.rand(2, 1, 4, 5, 1, 7, 8).astype(np.float32)
    d2 = np.random.rand(2, 3, 4, 5, 6, 8, 9).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        o = builder.aiOnnx.matmul([i1, i2])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        out = np.matmul(d1, d2)
        return [out]

    op_tester.run(init_builder, reference)


def test_matmul_grouped_3(op_tester):
    d1 = np.random.rand(4, 5, 1, 7, 8).astype(np.float32)
    d2 = np.random.rand(2, 3, 1, 5, 6, 8, 9).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        o = builder.aiOnnx.matmul([i1, i2])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        out = np.matmul(d1, d2)
        return [out]

    op_tester.run(init_builder, reference)


def test_matmul_grouped_4(op_tester):
    d1 = np.random.rand(2, 1, 4, 5, 1, 7, 8).astype(np.float32)
    d2 = np.random.rand(4, 5, 6, 8, 9).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        o = builder.aiOnnx.matmul([i1, i2])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        out = np.matmul(d1, d2)
        return [out]

    op_tester.run(init_builder, reference)


def test_matmul_grouped_5(op_tester):
    d1 = np.random.rand(3, 3, 3).astype(np.float32)
    d2 = np.random.rand(3, 3, 4).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        o = builder.aiOnnx.matmul([i1, i2])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        out = np.matmul(d1, d2)
        return [out]

    op_tester.run(init_builder, reference)
