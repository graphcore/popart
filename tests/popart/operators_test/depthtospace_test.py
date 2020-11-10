# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
import torch
from op_tester import op_tester


def test_depthtospace0(op_tester):
    # create test data
    d1 = np.random.rand(1, 8, 2, 3).astype(np.float32)
    blocks = 2

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnxOpset11.depthtospace([i1],
                                               blocksize=blocks,
                                               mode="CRD")
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        tx = torch.tensor(d1)
        d_shape = tx.size()
        txr = torch.reshape(
            tx, (d_shape[0], d_shape[1] //
                 (blocks * blocks), blocks, blocks, d_shape[2], d_shape[3]))
        txrp = txr.permute(0, 1, 4, 2, 5, 3)
        out = torch.reshape(
            txrp,
            (d_shape[0], d_shape[1] //
             (blocks * blocks), d_shape[2] * blocks, d_shape[3] * blocks))
        return [out]

    op_tester.setPatterns(['DepthToSpaceOpPattern'],
                          enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'infer')


def test_depthtospace1(op_tester):
    # create test data
    d1 = np.random.rand(1, 8, 2, 3).astype(np.float32)
    blocks = 2

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnxOpset11.depthtospace([i1],
                                               blocksize=blocks,
                                               mode="DCR")
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        tx = torch.tensor(d1)
        d_shape = tx.size()
        txr = torch.reshape(tx, (d_shape[0], blocks, blocks, d_shape[1] //
                                 (blocks * blocks), d_shape[2], d_shape[3]))
        txrp = txr.permute(0, 3, 4, 1, 5, 2)
        out = torch.reshape(
            txrp,
            (d_shape[0], d_shape[1] //
             (blocks * blocks), d_shape[2] * blocks, d_shape[3] * blocks))
        return [out]

    op_tester.setPatterns(['DepthToSpaceOpPattern'],
                          enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'infer')


def test_depthtospace_grad0(op_tester):
    d1 = np.random.rand(1, 8, 2, 3).astype(np.float32)
    blocks = 2

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnxOpset11.depthtospace([i1],
                                               blocksize=blocks,
                                               mode="CRD")
        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + i1,
            popart.reservedGradientPrefix() + o,
        ]

    def reference(ref_data):
        tx = torch.tensor(d1, requires_grad=True)
        d_shape = tx.size()
        txr = torch.reshape(
            tx, (d_shape[0], d_shape[1] //
                 (blocks * blocks), blocks, blocks, d_shape[2], d_shape[3]))
        txrp = txr.permute(0, 1, 4, 2, 5, 3)
        out = torch.reshape(
            txrp,
            (d_shape[0], d_shape[1] //
             (blocks * blocks), d_shape[2] * blocks, d_shape[3] * blocks))
        d__o = ref_data.getOutputTensorGrad(0)
        out.backward(torch.tensor(d__o))
        return [out, tx.grad, None]

    op_tester.setPatterns(['DepthToSpaceOpPattern'],
                          enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'train')


def test_depthtospace_grad1(op_tester):
    d1 = np.random.rand(1, 8, 2, 3).astype(np.float32)
    blocks = 2

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnxOpset11.depthtospace([i1],
                                               blocksize=blocks,
                                               mode="DCR")
        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + i1,
            popart.reservedGradientPrefix() + o,
        ]

    def reference(ref_data):
        tx = torch.tensor(d1, requires_grad=True)
        d_shape = tx.size()
        txr = torch.reshape(tx, (d_shape[0], blocks, blocks, d_shape[1] //
                                 (blocks * blocks), d_shape[2], d_shape[3]))
        txrp = txr.permute(0, 3, 4, 1, 5, 2)
        out = torch.reshape(
            txrp,
            (d_shape[0], d_shape[1] //
             (blocks * blocks), d_shape[2] * blocks, d_shape[3] * blocks))
        d__o = ref_data.getOutputTensorGrad(0)
        out.backward(torch.tensor(d__o))
        return [out, tx.grad, None]

    op_tester.setPatterns(['DepthToSpaceOpPattern'],
                          enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'train')
