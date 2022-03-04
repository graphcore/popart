# Copyright (c) 2019 Graphcore Ltd. All rights reserved.

# Required to avoid "tensor is not callable pylint error:"
# pylint: disable=E1102
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import popart

# `import test_util` requires adding to sys.path
sys.path.append(Path(__file__).resolve().parent.parent)
import test_util as tu


def test_get_op_types():
    ops_public = popart.getSupportedOperations(False)
    assert (len(ops_public) > 0)

    ops_all = popart.getSupportedOperations(True)
    assert (len(ops_all) > 0)
    assert (len(ops_all) > len(ops_public))


def test_get_op_types_definition():
    ops_public = popart.getSupportedOperationsDefinition(False)
    assert (len(ops_public) > 0)

    for k, v in ops_public.items():
        print(k.domain, k.type, k.version)
        print(" Inputs:")
        for i in v.inputs:
            print("  ", i.name, i.supportedTensors)
        print(" Outputs:")
        for o in v.outputs:
            print("  ", o.name, o.supportedTensors)
        print(" Attributes:")
        for a, r in v.attributes.items():
            print("  ", a, r.supportedValuesRegex)

        print("")

    ops_all = popart.getSupportedOperationsDefinition(True)
    assert (len(ops_all) > 0)
    assert (len(ops_all) > len(ops_public))


def test_add(op_tester):
    d1 = np.random.rand(2).astype(np.float32)
    d2 = np.random.rand(2).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        o = builder.aiOnnx.add([i1, i2], "test_add")
        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + i1,
            popart.reservedGradientPrefix() + i2,
            popart.reservedGradientPrefix() + o
        ]

    def reference(ref_data):
        t1 = torch.tensor(d1, requires_grad=True)
        t2 = torch.tensor(d2, requires_grad=True)
        out = t1 + t2

        d__o = torch.tensor(ref_data.getOutputTensorGrad(0))
        assert not torch.isnan(d__o).any()
        out.backward(d__o)

        return [out, t1.grad, t2.grad, None]

    op_tester.setPatterns(['PreUniRepl'], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'train')


def test_logical_if(op_tester):
    def test(then_branch):
        d1 = np.asarray([0, 20, 5]).astype(np.int32)
        d2 = np.asarray([7, 2, -1]).astype(np.int32)
        d3 = np.asarray(then_branch)

        def init_builder(builder):
            i1 = builder.addInputTensor(d1)
            i2 = builder.addInputTensor(d2)
            condition = builder.addInputTensor(d3)

            then_builder = builder.createSubgraphBuilder()
            then_builder.addInputTensorFromParentGraph(i1)
            then_builder.addInputTensorFromParentGraph(i2)
            then_builder.addOutputTensor(then_builder.aiOnnx.add([i1, i2]))

            else_builder = builder.createSubgraphBuilder()
            else_builder.addInputTensorFromParentGraph(i1)
            else_builder.addInputTensorFromParentGraph(i2)
            else_builder.addOutputTensor(else_builder.aiOnnx.sub([i1, i2]))

            o = builder.aiOnnx.logical_if([condition], 1, else_builder,
                                          then_builder)[0]
            builder.addOutputTensor(o)
            return [o]

        def reference(_):  # ref_data is an unused argument
            if then_branch is True:
                return [np.asarray([7, 22, 4]).astype(np.int32)]
            else:
                return [np.asarray([-7, 18, 6]).astype(np.int32)]

        op_tester.run(init_builder, reference, 'infer')

    test(True)
    test(False)


# Test else subgraph has no input from parent graph.
def test_logical_if_2(op_tester):
    def test(then_branch):
        d1 = np.asarray(1).astype(np.int32)
        d2 = np.asarray(2).astype(np.int32)
        d3 = np.asarray(then_branch)
        d_emptyConst = np.asarray(10).astype(np.int32)

        def init_builder(builder):
            i1 = builder.addInitializedInputTensor(d1)
            i2 = builder.aiOnnx.constant(d2)
            condition = builder.aiOnnx.constant(d3)

            then_builder = builder.createSubgraphBuilder()
            then_builder.addInputTensorFromParentGraph(i1)
            then_builder.addInputTensorFromParentGraph(i2)
            then_builder.addOutputTensor(then_builder.aiOnnx.add([i1, i2]))

            else_builder = builder.createSubgraphBuilder()
            d_emptyOut = else_builder.aiOnnx.constant(d_emptyConst)
            d_out = else_builder.aiGraphcore.nop([d_emptyOut])
            else_builder.addOutputTensor(d_out)
            o = builder.aiOnnx.logical_if([condition], 1, else_builder,
                                          then_builder)[0]
            builder.addOutputTensor(o)
            return [o]

        def reference(_):  # ref_data is an unused argument
            if then_branch is True:
                return [np.asarray(3).astype(np.int32)]
            else:
                return [np.asarray(10).astype(np.int32)]

        op_tester.run(init_builder, reference, 'infer')

    test(True)
    test(False)


def test_loop(op_tester):
    i1 = np.array([[1, 2], [3, 4]]).astype(np.float32)
    i2 = np.array([[1, 2], [3, 4]]).astype(np.float32)
    trip_count = 10

    def init_builder(builder):
        builder.setGraphName("main_graph")
        a = builder.addInputTensor(i1)
        b = builder.addInputTensor(i2)

        M = builder.aiOnnx.constant(np.array(trip_count).astype(np.int64))
        cond = builder.aiOnnx.constant(np.array(True).astype(np.bool), "cond")

        loop_builder = builder.createSubgraphBuilder()
        loop_builder.setGraphName("body")
        # Inputs: [iteration_number, condition_in, a_in]
        loop_builder.addInputTensor(popart.TensorInfo("INT64", []))
        keepgoing = loop_builder.addInputTensor(popart.TensorInfo("BOOL", []))
        a_in = loop_builder.addUntypedInputTensor(a)
        a_out = loop_builder.aiOnnx.matmul([a_in, b])

        # Outputs: [condition_out, a_out]
        loop_builder.addOutputTensor(keepgoing)
        loop_builder.addOutputTensor(a_out)

        o = builder.aiOnnx.loop([M, cond, a], 1, loop_builder)[0]
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        a = i1
        b = i2

        x = a
        for i in range(trip_count):
            x = np.matmul(x, b)

        return [x]

    op_tester.run(init_builder, reference, step_type='infer')


def test_convolution(op_tester):
    def init_builder(builder):
        data = np.ones([1, 2, 4, 4], dtype=np.float32)
        filt = np.ones([3, 2, 3, 3], dtype=np.float32)
        d = builder.addInputTensor(data)
        f = builder.addInputTensor(filt)
        o = builder.aiOnnx.conv([d, f],
                                dilations=[1, 1],
                                pads=[1, 1, 1, 1],
                                strides=[1, 1])
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        expected = np.array([[[[8., 12., 12., 8.], [12., 18., 18., 12.],
                               [12., 18., 18., 12.], [8., 12., 12., 8.]],
                              [[8., 12., 12., 8.], [12., 18., 18., 12.],
                               [12., 18., 18., 12.], [8., 12., 12., 8.]],
                              [[8., 12., 12., 8.], [12., 18., 18., 12.],
                               [12., 18., 18., 12.], [8., 12., 12., 8.]]]],
                            dtype=np.float32)
        return [expected]

    op_tester.run(init_builder, reference, step_type='infer')


def test_convolution_2(op_tester):
    '''
    Test the convolution when the conv in the bwd pass is not the same as the conv in the
    forward pass
    '''

    def init_builder(builder):
        data = np.ones([1, 2, 4, 4], dtype=np.float32)
        filt = np.ones([4, 2, 1, 1], dtype=np.float32)
        d = builder.addInputTensor(data)
        f = builder.addInputTensor(filt)
        o = builder.aiOnnx.conv([d, f],
                                dilations=[1, 1],
                                pads=[0, 0, 0, 0],
                                strides=[2, 2])
        builder.addOutputTensor(o)
        return [o, popart.reservedGradientPrefix() + d]

    def reference(_):  # ref_data is an unused argument
        expected = np.array([[[[2., 2.], [2., 2.]], [[2., 2.], [2., 2.]],
                              [[2., 2.], [2., 2.]], [[2., 2.], [2., 2.]]]],
                            dtype=np.float32)
        return [expected, None]

    op_tester.setPatterns(["ConvDataGrad"], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, step_type='train')


def test_convolution_3(op_tester):
    batch_size = 1
    chans_in = 2
    chans_out = 3
    size = 4
    kernel_size = 3
    padding = 1

    data = np.ones([batch_size, chans_in, size, size], dtype=np.float32)
    filt = np.ones([chans_out, chans_in, kernel_size, kernel_size],
                   dtype=np.float32)

    def init_builder(builder):
        d = builder.addInputTensor(data)
        f = builder.addInputTensor(filt)
        o = builder.aiOnnx.conv([d, f],
                                dilations=[1, 1],
                                pads=[padding] * 4,
                                strides=[1, 1])
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        d = torch.tensor(data)
        conv = torch.nn.Conv2d(chans_in,
                               chans_out,
                               kernel_size,
                               padding=[padding] * 2)
        conv.weight.data = torch.tensor(filt)
        conv.bias.data = torch.tensor([0.0 for i in range(chans_out)])
        o = conv(d)
        return [o]

    op_tester.run(init_builder, reference, step_type='infer')


def test_convolution_4(op_tester):
    batch_size = 1
    chans_in = 6
    chans_out = 9
    size = 4
    kernel_size = 3
    padding = 1
    groups = 3

    data = np.random.rand(batch_size, chans_in, size, size).astype(np.float32)

    filt = np.random.rand(chans_out, chans_in // groups, kernel_size,
                          kernel_size).astype(np.float32)

    def init_builder(builder):
        d = builder.addInputTensor(data)
        f = builder.addInputTensor(filt)
        o = builder.aiOnnx.conv([d, f],
                                dilations=[1, 1],
                                pads=[padding] * 4,
                                strides=[1, 1],
                                group=groups)
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        d = torch.tensor(data)
        conv = torch.nn.Conv2d(chans_in,
                               chans_out,
                               kernel_size,
                               padding=[padding] * 2,
                               groups=groups)
        conv.weight.data = torch.tensor(filt)
        conv.bias.data = torch.tensor([0.0 for i in range(chans_out)])
        o = conv(d)
        return [o]

    op_tester.run(init_builder, reference, step_type='infer')


def test_convolution_5(op_tester):
    batch_size = 1
    size = 4
    kernel_size = 3
    padding = 1
    groups = 5
    # chans_in/out must be divisible by groups
    chans_in = groups * 11
    chans_out = groups * 7

    data = np.random.rand(batch_size, chans_in, size, size).astype(np.float32)

    filt = np.random.rand(chans_out, chans_in // groups, kernel_size,
                          kernel_size).astype(np.float32)

    def init_builder0(builder):
        d = builder.addInputTensor(data)
        f = builder.addInputTensor(filt)
        o = builder.aiOnnx.conv([d, f],
                                dilations=[1, 1],
                                pads=[padding] * 4,
                                strides=[1, 1],
                                group=groups)
        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + d,
            popart.reservedGradientPrefix() + o
        ]

    def init_builder1(builder):
        d = builder.addInputTensor(data)
        f = builder.addInitializedInputTensor(filt)
        o = builder.aiOnnx.conv([d, f],
                                dilations=[1, 1],
                                pads=[padding] * 4,
                                strides=[1, 1],
                                group=groups)
        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + d,
            popart.reservedGradientPrefix() + o
        ]

    def reference(ref_data):
        d = torch.tensor(data, requires_grad=True)
        conv = torch.nn.Conv2d(chans_in,
                               chans_out,
                               kernel_size,
                               padding=[padding] * 2,
                               groups=groups)
        conv.weight.data = torch.tensor(filt)
        conv.bias.data = torch.tensor([0.0 for i in range(chans_out)])
        o = conv(d)
        d__o = ref_data.getOutputTensorGrad(0)
        o.backward(torch.tensor(d__o))
        dg = d.grad

        return [o, dg, None]

    op_tester.setPatterns(['ConvDataGrad'], enableRuntimeAsserts=False)
    op_tester.run(init_builder0, reference, step_type='train')
    op_tester.run(init_builder1, reference, step_type='train')


def test_convolution_6(op_tester):
    batch_size = 1
    chans_in = 2
    chans_out = 3
    size = 4
    kernel_size = 3
    padding = 5  # Deliberately excessive

    data = np.ones([batch_size, chans_in, size, size], dtype=np.float32)
    filt = np.ones([chans_out, chans_in, kernel_size, kernel_size],
                   dtype=np.float32)

    def init_builder(builder):
        d = builder.addInputTensor(data)
        f = builder.addInputTensor(filt)
        o = builder.aiOnnx.conv([d, f],
                                dilations=[1, 1],
                                pads=[padding] * 4,
                                strides=[1, 1])
        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + d,
            popart.reservedGradientPrefix() + o
        ]

    def reference(ref_data):
        d = torch.tensor(data)
        conv = torch.nn.Conv2d(chans_in,
                               chans_out,
                               kernel_size,
                               padding=[padding] * 2)
        conv.weight.data = torch.tensor(filt)
        conv.bias.data = torch.tensor([0.0 for i in range(chans_out)])
        o = conv(d)
        d__o = ref_data.getOutputTensorGrad(0)
        o.backward(torch.tensor(d__o))
        dg = d.grad

        return [o, dg, None]

    op_tester.setPatterns(['ConvDataGrad'], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, step_type='train')


def test_convolution_7(op_tester):
    batch_size = 1
    chans_in = 4
    chans_out = 2
    size_y = 8
    size_x = 10
    kernel_size_y = 3
    kernel_size_x = 5
    padding_y = 4
    padding_x = 2
    dilation_y = 3
    dilation_x = 3
    stride_y = 2
    stride_x = 1

    data = np.ones([batch_size, chans_in, size_y, size_x], dtype=np.float32)
    filt = np.ones([chans_out, chans_in, kernel_size_y, kernel_size_x],
                   dtype=np.float32)

    def init_builder(builder):
        d = builder.addInputTensor(data)
        f = builder.addInputTensor(filt)
        o = builder.aiOnnx.conv(
            [d, f],
            dilations=[dilation_y, dilation_x],
            pads=[padding_y, padding_x, padding_y, padding_x],
            strides=[stride_y, stride_x])
        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + d,
            popart.reservedGradientPrefix() + o
        ]

    def reference(ref_data):
        d = torch.tensor(data, requires_grad=True)
        conv = torch.nn.Conv2d(chans_in,
                               chans_out, (kernel_size_y, kernel_size_x),
                               stride=(stride_y, stride_x),
                               padding=(padding_y, padding_x),
                               dilation=(dilation_y, dilation_x))
        conv.weight.data = torch.tensor(filt)
        conv.bias.data = torch.tensor([0.0 for i in range(chans_out)])
        o = conv(d)

        # With dilation, the input kernel size is 7 by 13:
        # X~~X~~X~~X~~X
        # ~~~~~~~~~~~~~
        # ~~~~~~~~~~~~~
        # X~~X~~X~~X~~X
        # ~~~~~~~~~~~~~
        # ~~~~~~~~~~~~~
        # X~~X~~X~~X~~X

        # The padded input spatial size is (16, 14).
        # For the forward convolution, the spatial output size is (5, 2).
        # This means that the left and right of the kernel onlys convolve over
        # the padded input. The data gradient, therefore, is only affected by
        # the middle column of the kernel.

        # Poplin identifies this when calculating the backwards conv parameters
        # and truncates the kernel on both sides of the x-axis. This test tests
        # that the truncation is correctlu passed to the ConvOp created to
        # replace the ConvDataGradOp.

        d__o = ref_data.getOutputTensorGrad(0)
        o.backward(torch.tensor(d__o))
        dg = d.grad

        return [o, dg, None]

    op_tester.setPatterns(['ConvDataGrad'], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, step_type='train')


def test_convolution_3d(op_tester):
    batch_size = 1
    chans_in = 2
    chans_out = 3
    size = 4
    kernel_size = 3
    padding = 1

    data = np.ones([batch_size, chans_in, size, size, size], dtype=np.float32)
    filt = np.ones(
        [chans_out, chans_in, kernel_size, kernel_size, kernel_size],
        dtype=np.float32)

    def init_builder(builder):
        d = builder.addInputTensor(data)
        f = builder.addInputTensor(filt)
        o = builder.aiOnnx.conv([d, f],
                                dilations=[1, 1, 1],
                                pads=[padding] * 6,
                                strides=[1, 1, 1])
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        d = torch.tensor(data)
        conv = torch.nn.Conv3d(chans_in,
                               chans_out,
                               kernel_size,
                               padding=[padding] * 3)
        conv.weight.data = torch.tensor(filt)
        conv.bias.data = torch.tensor([0.0 for i in range(chans_out)])
        o = conv(d)
        return [o]

    op_tester.run(init_builder, reference, step_type='infer')


def test_convolution_3d_2(op_tester):
    batch_size = 1
    size = 4
    kernel_size = 3
    padding = 1
    groups = 5
    # chans_in/out must be divisible by groups
    chans_in = groups * 11
    chans_out = groups * 7

    data = np.random.rand(batch_size, chans_in, size, size,
                          size).astype(np.float32)

    filt = np.random.rand(chans_out, chans_in // groups, kernel_size,
                          kernel_size, kernel_size).astype(np.float32)

    def init_builder0(builder):
        d = builder.addInputTensor(data)
        f = builder.addInputTensor(filt)
        o = builder.aiOnnx.conv([d, f],
                                dilations=[1, 1, 1],
                                pads=[padding] * 6,
                                strides=[1, 1, 1],
                                group=groups)
        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + d,
            popart.reservedGradientPrefix() + o
        ]

    def init_builder1(builder):
        d = builder.addInputTensor(data)
        f = builder.addInitializedInputTensor(filt)
        o = builder.aiOnnx.conv([d, f],
                                dilations=[1, 1, 1],
                                pads=[padding] * 6,
                                strides=[1, 1, 1],
                                group=groups)
        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + d,
            popart.reservedGradientPrefix() + o
        ]

    def reference(ref_data):
        d = torch.tensor(data, requires_grad=True)
        conv = torch.nn.Conv3d(chans_in,
                               chans_out,
                               kernel_size,
                               padding=[padding] * 3,
                               groups=groups)
        conv.weight.data = torch.tensor(filt)
        conv.bias.data = torch.tensor([0.0 for i in range(chans_out)])
        o = conv(d)
        d__o = ref_data.getOutputTensorGrad(0)
        o.backward(torch.tensor(d__o))
        dg = d.grad

        return [o, dg, None]

    op_tester.setPatterns(['ConvDataGrad'], enableRuntimeAsserts=False)
    op_tester.run(init_builder0, reference, step_type='train')
    op_tester.run(init_builder1, reference, step_type='train')


def test_convolution_default_infer(op_tester):
    batch_size = 1
    chans_in = 2
    chans_out = 3
    size = 4
    kernel_size = 3

    data = np.random.rand(batch_size, chans_in, size, size).astype(np.float32)
    filt = np.random.rand(chans_out, chans_in, kernel_size,
                          kernel_size).astype(np.float32)

    def init_builder(builder):
        d = builder.addInputTensor(data)
        f = builder.addInputTensor(filt)
        o = builder.aiOnnx.conv([d, f])
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        d = torch.tensor(data)
        conv = torch.nn.Conv2d(chans_in, chans_out, kernel_size)
        conv.weight.data = torch.tensor(filt)
        conv.bias.data = torch.tensor([0.0 for i in range(chans_out)])
        o = conv(d)
        return [o]

    op_tester.run(init_builder, reference, step_type='infer')


def test_convolution_default_train(op_tester):
    batch_size = 1
    chans_in = 2
    chans_out = 3
    size = 4
    kernel_size = 3

    data = np.random.rand(batch_size, chans_in, size, size).astype(np.float32)
    filt = np.random.rand(chans_out, chans_in, kernel_size,
                          kernel_size).astype(np.float32)

    def init_builder(builder):
        d = builder.addInputTensor(data)
        f = builder.addInputTensor(filt)
        o = builder.aiOnnx.conv([d, f])
        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + d,
            popart.reservedGradientPrefix() + o
        ]

    def reference(ref_data):
        d = torch.tensor(data)
        conv = torch.nn.Conv2d(chans_in, chans_out, kernel_size)
        conv.weight.data = torch.tensor(filt)
        conv.bias.data = torch.tensor([0.0 for i in range(chans_out)])
        o = conv(d)
        d__o = ref_data.getOutputTensorGrad(0)
        o.backward(torch.tensor(d__o))
        dg = d.grad

        return [o, dg, None]

    op_tester.setPatterns(['ConvDataGrad'], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, step_type='train')


def test_convolution_with_bias_1d(op_tester):
    batch_size = 1
    chans_in = 2
    chans_out = 3
    size = 4
    kernel_size = 3
    padding = 1

    data = np.ones([batch_size, chans_in, size], dtype=np.float32)
    filt = np.ones([chans_out, chans_in, kernel_size], dtype=np.float32)
    bias = np.arange(chans_out, dtype=np.float32)

    def init_builder(builder):
        d = builder.addInputTensor(data)
        f = builder.addInputTensor(filt)
        b = builder.addInputTensor(bias)
        o = builder.aiOnnx.conv([d, f, b],
                                dilations=[1],
                                pads=[padding] * 2,
                                strides=[1])
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        d = torch.tensor(data)
        conv = torch.nn.Conv1d(chans_in,
                               chans_out,
                               kernel_size,
                               padding=[padding])
        conv.weight.data = torch.tensor(filt)
        conv.bias.data = torch.tensor(bias)
        o = conv(d)
        return [o]

    op_tester.run(init_builder, reference, step_type='infer')


def test_reciprocal(op_tester):
    # create test data
    d1 = np.random.rand(4).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.reciprocal([i1])
        builder.addOutputTensor(o)
        return [o]

    # create and run numpy reference
    def reference(_):  # ref_data is an unused argument
        return [1 / d1]

    op_tester.run(init_builder, reference)


def test_div(op_tester):
    d1 = np.random.rand(4).astype(np.float32)
    d2 = np.random.rand(4).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        o = builder.aiOnnx.div([i1, i2], "test_div")
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        out = d1 / d2
        return [out]

    op_tester.setPatterns(['PreUniRepl'], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'infer')


def test_div_grad(op_tester):
    d1 = np.random.rand(4).astype(np.float32)
    d2 = np.random.rand(4).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        o = builder.aiOnnx.div([i1, i2], "test_div")
        builder.addOutputTensor(o)
        gradPrefix = popart.reservedGradientPrefix()
        return [o, gradPrefix + i1, gradPrefix + i2, gradPrefix + o]

    def reference(ref_data):
        t1 = torch.tensor(d1, requires_grad=True)
        t2 = torch.tensor(d2, requires_grad=True)
        out = t1 / t2

        d__o = torch.tensor(ref_data.getOutputTensorGrad(0))
        assert not torch.isnan(d__o).any()
        out.backward(d__o)

        return [out, t1.grad, t2.grad, None]

    op_tester.setPatterns(['PreUniRepl', 'DivArg0GradOp', 'DivArg1GradOp'],
                          enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'train')


def test_fmod(op_tester):
    d1 = np.random.rand(4).astype(np.float32) * 10.0
    d2 = np.random.rand(4).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        o = builder.aiGraphcore.fmod([i1, i2], "test_fmod")
        # Check builder shape inference
        assert (builder.getTensorShape(o) == [4])
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        out = np.fmod(d1, d2)
        return [out]

    op_tester.run(init_builder, reference, 'infer')


def test_fmod_grad(op_tester):
    d1 = np.random.rand(4).astype(np.float32) * 10.0
    d2 = np.random.rand(4).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        o = builder.aiGraphcore.fmod([i1, i2], "test_fmod")
        builder.addOutputTensor(o)
        gradPrefix = popart.reservedGradientPrefix()

        return [o, gradPrefix + i1, i2, gradPrefix + o]

    def reference(ref_data):
        t1 = torch.tensor(d1, requires_grad=True)
        t2 = torch.tensor(d2, requires_grad=False)
        out = torch.fmod(t1, t2)
        d__o = torch.tensor(ref_data.getOutputTensorGrad(0))
        assert not torch.isnan(d__o).any()
        out.backward(d__o)

        return [out, t1.grad, t2, None]

    op_tester.setPatterns(['FmodArg0GradOp'], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'train')


def test_fmod_mixed_sign_float16(op_tester):
    d1 = np.array([-4.3, 7.2, 5.0, 4.3, -7.2, 8.0]).astype(np.float16)
    d2 = np.array([2.1, -3.4, 8.0, -2.1, 3.4, 5.0]).astype(np.float16)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        o = builder.aiGraphcore.fmod([i1, i2], "test_fmod")
        # Check builder shape inference
        assert (builder.getTensorShape(o) == [6])
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        out = np.fmod(d1, d2)
        return [out]

    op_tester.run(init_builder, reference, 'infer')


def test_remainder_grad(op_tester):
    d1 = (np.random.rand(4).astype(np.float32) - 0.5) * 10.0
    d2 = np.random.rand(4).astype(np.float32) - 0.5

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        o = builder.aiGraphcore.remainder([i1, i2], "test_remainder_grad")
        builder.addOutputTensor(o)
        gradPrefix = popart.reservedGradientPrefix()
        # Check builder shape inference
        assert (builder.getTensorShape(o) == [4])

        return [o, gradPrefix + i1, i2, gradPrefix + o]

    def reference(ref_data):
        t1 = torch.tensor(d1, requires_grad=True)
        t2 = torch.tensor(d2, requires_grad=False)
        out = torch.remainder(t1, t2)
        d__o = torch.tensor(ref_data.getOutputTensorGrad(0))
        assert not torch.isnan(d__o).any()
        out.backward(torch.tensor(d__o))

        return [out, t1.grad, t2, None]

    op_tester.setPatterns(['FmodArg0GradOp'], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'train')


@pytest.mark.parametrize("fmod_attr", [0, 1])
def test_onnx_mod_grad(op_tester, fmod_attr):
    d1 = (np.random.rand(4).astype(np.float32) - 0.5) * 10.0
    d2 = np.random.rand(4).astype(np.float32) - 0.5

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        o = builder.aiOnnx.mod([i1, i2], fmod_attr, "test_onnx_mod_grad")
        builder.addOutputTensor(o)
        gradPrefix = popart.reservedGradientPrefix()

        return [o, gradPrefix + i1, i2, gradPrefix + o]

    def reference(ref_data):
        t1 = torch.tensor(d1, requires_grad=True)
        t2 = torch.tensor(d2, requires_grad=False)
        out = torch.remainder(t1, t2) if fmod_attr == 0 else torch.fmod(t1, t2)
        d__o = torch.tensor(ref_data.getOutputTensorGrad(0))
        assert not torch.isnan(d__o).any()
        out.backward(torch.tensor(d__o))

        return [out, t1.grad, t2, None]

    op_tester.setPatterns(['FmodArg0GradOp'], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'train')


@pytest.mark.parametrize("fmod_attr", [0, 1])
def test_mod_mixed_sign_float16(op_tester, fmod_attr):
    d1 = np.array([-4.3, 7.2, 5.0, 4.3, -7.2, 8.0]).astype(np.float16)
    d2 = np.array([2.1, -3.4, 8.0, -2.1, 3.4, 5.0]).astype(np.float16)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        o = builder.aiOnnx.mod([i1, i2], fmod_attr, "test_mod")
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        out = np.fmod(d1, d2) if fmod_attr == 1 else np.mod(d1, d2)
        return [out]

    op_tester.run(init_builder, reference, 'infer')


def test_reciprocal_grad(op_tester):
    # create test data
    d1 = np.random.rand(4).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.reciprocal([i1])
        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + i1,
            popart.reservedGradientPrefix() + o
        ]

    def reference(ref_data):
        a = torch.tensor(d1, requires_grad=True)
        b = 1 / a
        d__o = ref_data.getOutputTensorGrad(0)
        b.backward(torch.tensor(d__o))
        return [b, a.grad, None]

    op_tester.setPatterns(['PreUniRepl', 'ReciprocalGradOp'],
                          enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'train')


def test_reverse(op_tester):
    d1 = np.random.rand(4, 1, 3).astype(np.float32)
    reverse_dims = [2, 1]

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiGraphcore.reverse([i1], reverse_dims)
        builder.addOutputTensor(o)
        return [o]

    def init_builder_negative_dim(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiGraphcore.reverse([i1], [2, -2])
        return [o]

    def init_builder_dim_appears_twice(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiGraphcore.reverse([i1], [1, 1])
        return [o]

    def init_builder_dim_greater_than_rank(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiGraphcore.reverse([i1], [3])
        return [o]

    def reference(_):  # ref_data is an unused argument
        out = torch.flip(torch.tensor(d1), reverse_dims)
        return [out]

    op_tester.run(init_builder, reference, 'infer')

    with pytest.raises(popart.popart_exception) as e_info:
        op_tester.run(init_builder_negative_dim, reference, 'infer')
    assert "invalid dimension '-2'. Only positive dimensions are supported" in e_info.value.args[
        0]

    with pytest.raises(popart.popart_exception) as e_info:
        op_tester.run(init_builder_dim_appears_twice, reference, 'infer')
    assert "Dimension 1 appears multiple times" in e_info.value.args[0]

    with pytest.raises(popart.popart_exception) as e_info:
        op_tester.run(init_builder_dim_greater_than_rank, reference, 'infer')
    assert "invalid dimension '3' for input tensor of rank 3" in e_info.value.args[
        0]


def test_reverse_grad(op_tester):
    d1 = np.random.rand(4, 1, 3).astype(np.float32)
    reverse_dims = [2, 1]

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiGraphcore.reverse([i1], reverse_dims)
        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + i1,
            popart.reservedGradientPrefix() + o
        ]

    def reference(ref_data):
        a = torch.tensor(d1, requires_grad=True)
        b = torch.flip(torch.tensor(d1), reverse_dims)
        d__o = ref_data.getOutputTensorGrad(0)
        return [b, torch.flip(torch.tensor(d__o), reverse_dims), None]

    op_tester.run(init_builder, reference, 'train')


def test_sqrt(op_tester):
    # create test data
    d1 = np.random.rand(4).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.sqrt([i1])
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        a = torch.tensor(d1, requires_grad=True)
        out = torch.sqrt(a)
        return [out]

    op_tester.setPatterns(['PreUniRepl'], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'infer')


def test_sqrt_grad(op_tester):
    # create test data
    d1 = np.random.rand(4).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.sqrt([i1])
        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + i1,
            popart.reservedGradientPrefix() + o
        ]

    def reference(ref_data):
        a = torch.tensor(d1, requires_grad=True)
        out = torch.sqrt(a)
        d__o = ref_data.getOutputTensorGrad(0)
        out.backward(torch.tensor(d__o))
        return [out, a.grad, None]

    op_tester.setPatterns(['PreUniRepl', 'SqrtGradOp'],
                          enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'train')


def test_subtract(op_tester):
    d1 = np.random.rand(4).astype(np.float32)
    d2 = np.random.rand(4).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        o = builder.aiOnnx.sub([i1, i2], "test_subtract")
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        out = d1 - d2
        return [out]

    op_tester.setPatterns(['PreUniRepl'], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'infer')


def test_subtract_grad(op_tester):
    d1 = np.random.rand(4).astype(np.float32)
    d2 = np.random.rand(4).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        o = builder.aiOnnx.sub([i1, i2], "test_subtract")
        builder.addOutputTensor(o)
        gradPrefix = popart.reservedGradientPrefix()
        return [o, gradPrefix + i1, gradPrefix + i2, gradPrefix + o]

    def reference(ref_data):
        t1 = torch.tensor(d1, requires_grad=True)
        t2 = torch.tensor(d2, requires_grad=True)
        out = t1 - t2

        d__o = torch.tensor(ref_data.getOutputTensorGrad(0))
        assert not torch.isnan(d__o).any()
        out.backward(d__o)

        return [out, t1.grad, t2.grad, None]

    op_tester.setPatterns(['PreUniRepl', 'SubtractArg1GradOp'],
                          enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'train')


def test_exp(op_tester):
    # create test data
    d1 = np.random.rand(4).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.exp([i1])
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        a = torch.tensor(d1, requires_grad=True)
        b = torch.exp(a)
        return [b]

    op_tester.run(init_builder, reference, 'infer')


def test_exp_grad(op_tester):
    # create test data
    d1 = np.random.rand(4).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.exp([i1])
        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + i1,
            popart.reservedGradientPrefix() + o
        ]

    def reference(ref_data):
        a = torch.tensor(d1, requires_grad=True)
        b = torch.exp(a)
        d__o = ref_data.getOutputTensorGrad(0)
        b.backward(torch.tensor(d__o))
        return [b, a.grad, None]

    op_tester.setPatterns(['PreUniRepl', 'ExpGradOp'],
                          enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'train')


def test_sigmoid(op_tester):
    # create test data
    d1 = np.random.rand(4).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.sigmoid([i1])
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        a = torch.tensor(d1, requires_grad=True)
        b = torch.sigmoid(a)
        return [b]

    op_tester.run(init_builder, reference, 'infer')


def test_sigmoid_grad(op_tester):
    # create test data
    d1 = np.random.rand(4).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.sigmoid([i1])
        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + i1,
            popart.reservedGradientPrefix() + o
        ]

    def reference(ref_data):
        a = torch.tensor(d1, requires_grad=True)
        b = torch.sigmoid(a)
        d__o = ref_data.getOutputTensorGrad(0)
        b.backward(torch.tensor(d__o))
        return [b, a.grad, None]

    op_tester.setPatterns(['PreUniRepl'], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'train')


def test_topk_2d(op_tester):
    d1 = np.random.rand(7, 8).astype(np.float32) * 10
    k = 4
    for axis in [0, 1]:

        def init_builder(builder):
            i1 = builder.addInputTensor(d1)
            k_t = builder.aiOnnx.constant(np.array([k]).astype(np.int64))
            [vals, inds] = builder.aiOnnx.topk([i1, k_t], axis=axis)
            builder.addOutputTensor(vals)
            return [vals, inds]

        def reference(_):  # ref_data is an unused argument
            a = torch.tensor(d1)
            b = torch.topk(a, k=k, dim=axis)
            return [b.values, b.indices]

        # Torch doesn't have a uint32 type
        op_tester.check_dtypes = False
        op_tester.run(init_builder, reference, 'infer')


def test_topk_2d_smallest(op_tester):
    d1 = np.random.rand(7, 8).astype(np.float32) * 10
    k = 4
    for axis in [0, 1]:

        def init_builder(builder):
            i1 = builder.addInputTensor(d1)
            k_t = builder.aiOnnx.constant(np.array([k]).astype(np.int64))
            [vals, inds] = builder.aiOnnx.topk([i1, k_t], axis=axis, largest=0)

            builder.addOutputTensor(vals)
            return [vals, inds]

        def reference(_):  # ref_data is an unused argument
            a = torch.tensor(d1)
            b = torch.topk(a, k=k, dim=axis, largest=False)
            return [b.values, b.indices]

        # Torch doesn't have a uint32 type
        op_tester.check_dtypes = False
        op_tester.run(init_builder,
                      reference,
                      'infer',
                      opsets={
                          "ai.onnx": 11,
                          "ai.graphcore": 1
                      })


def test_topk_2d_sorted():
    np.random.seed(0)
    d1 = np.random.rand(7, 8).astype(np.float32) * 10
    k = 4

    def run_test(sort_topk):
        if sort_topk:
            sort_topk = 1
        else:
            sort_topk = 0

        bld = popart.Builder(opsets={"ai.onnx": 11, "ai.graphcore": 1})
        i0 = bld.addInputTensor(popart.TensorInfo("FLOAT", [7, 8]))
        k_t = bld.aiOnnx.constant(np.array([k]).astype(np.int64))
        [vals, inds] = bld.aiOnnx.topk([i0, k_t], axis=0, sorted=sort_topk)

        bld.addOutputTensor(vals)

        with tu.create_test_device() as device:
            sess = popart.InferenceSession(bld.getModelProto(),
                                           deviceInfo=device,
                                           dataFlow=popart.DataFlow(1, [vals]))

            sess.prepareDevice()
            anchors = sess.initAnchorArrays()
            stepio = popart.PyStepIO({i0: d1}, anchors)
            sess.run(stepio)
        return anchors[vals]

    sorted_output = run_test(True)
    unsorted_output = run_test(False)

    # The values should not be equal, as one should
    # be sorted and the other should not.
    assert not np.allclose(sorted_output, unsorted_output)

    # The sums of the values should be equal, as they should
    # be the same values.
    assert np.isclose(np.sum(sorted_output), np.sum(unsorted_output))


def test_topk_2d_grad(op_tester):
    d1 = np.random.rand(7, 8).astype(np.float32) * 10
    k = 4
    for axis in [0, 1]:

        def init_builder(builder):
            i1 = builder.addInputTensor(d1)
            k_t = builder.aiOnnx.constant(np.array([k]).astype(np.int64))
            [vals, inds] = builder.aiOnnx.topk([i1, k_t], axis=axis)
            builder.addOutputTensor(vals)
            return [
                vals, inds,
                popart.reservedGradientPrefix() + i1,
                popart.reservedGradientPrefix() + vals
            ]

        def reference(ref_data):
            a = torch.tensor(d1, requires_grad=True)
            b = torch.topk(a, k=k, dim=axis)
            d__o = ref_data.getOutputTensorGrad(0)
            b.values.backward(torch.tensor(d__o))
            return [b.values, b.indices, a.grad, None]

        # Torch doesn't have a uint32 type
        op_tester.check_dtypes = False
        op_tester.run(init_builder, reference, 'train')


def test_topk_2d_smallest_grad(op_tester):
    d1 = np.random.rand(7, 8).astype(np.float32) * 10
    k = 4
    for axis in [0, 1]:

        def init_builder(builder):
            i1 = builder.addInputTensor(d1)
            k_t = builder.aiOnnx.constant(np.array([k]).astype(np.int64))
            [vals, inds] = builder.aiOnnx.topk([i1, k_t], axis=axis, largest=0)

            builder.addOutputTensor(vals)
            return [
                vals, inds,
                popart.reservedGradientPrefix() + i1,
                popart.reservedGradientPrefix() + vals
            ]

        def reference(ref_data):
            a = torch.tensor(d1, requires_grad=True)
            b = torch.topk(a, k=k, dim=axis, largest=False)
            d__o = ref_data.getOutputTensorGrad(0)
            b.values.backward(torch.tensor(d__o))
            return [b.values, b.indices, a.grad, None]

        # Torch doesn't have a uint32 type
        op_tester.check_dtypes = False
        op_tester.run(init_builder,
                      reference,
                      'train',
                      opsets={
                          "ai.onnx": 11,
                          "ai.graphcore": 1
                      })


def test_topk_2d_unsorted_grad(op_tester):
    d1 = np.random.rand(7, 8).astype(np.float32) * 10
    k = 4
    # for axis in [0, 1]:
    for axis in [0]:

        def init_builder(builder):
            i1 = builder.addInputTensor(d1)
            k_t = builder.aiOnnx.constant(np.array([k]).astype(np.int64))
            [vals, inds] = builder.aiOnnx.topk([i1, k_t], axis=axis, sorted=0)

            builder.addOutputTensor(vals)
            return [
                vals, inds,
                popart.reservedGradientPrefix() + i1,
                popart.reservedGradientPrefix() + vals
            ]

        def reference(ref_data):
            a = torch.tensor(d1, requires_grad=True)
            b = torch.topk(a, k=k, dim=axis, sorted=False)
            d__o = ref_data.getOutputTensorGrad(0)
            b.values.backward(torch.tensor(d__o))
            # Not comparing forward pass output.
            # Unsorted topk vals will not be equal.
            return [None, None, a.grad, None]

        # Torch doesn't have a uint32 type
        op_tester.check_dtypes = False
        op_tester.run(init_builder,
                      reference,
                      'train',
                      opsets={
                          "ai.onnx": 11,
                          "ai.graphcore": 1
                      })


def test_transpose(op_tester):
    d1 = np.random.rand(3, 5, 2, 7).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.transpose([i1], [2, 0, 3, 1])
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        a = np.transpose(d1, axes=[2, 0, 3, 1])
        return [a]

    op_tester.run(init_builder, reference, 'infer')


def test_transpose_grad(op_tester):
    d1 = np.random.rand(1, 3, 2, 7, 5).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.transpose([i1], [1, 3, 0, 4, 2])
        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + i1,
            popart.reservedGradientPrefix() + o
        ]

    def reference(ref_data):
        a = torch.tensor(d1, requires_grad=True)
        o = a.permute(1, 3, 0, 4, 2)

        d__o = ref_data.getOutputTensorGrad(0)
        o.backward(torch.tensor(d__o))

        return [o, a.grad, None]

    op_tester.setPatterns(['PreUniRepl'], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'train')


def test_transpose_sizes(op_tester):
    d1 = np.random.rand(1, 3, 2, 7, 5).astype(np.float32)
    transpose = [1, 3, 0, 4]

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.transpose([i1], transpose)
        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + i1,
            popart.reservedGradientPrefix() + o
        ]

    def reference(_):  # ref_data is an unused argument
        return []

    op_tester.setPatterns(['PreUniRepl'], enableRuntimeAsserts=False)
    with pytest.raises(popart.popart_exception) as e_info:
        op_tester.run(init_builder, reference, 'infer')

    assert (
        e_info.value.args[0] ==
        f"Rank of permutation tensor [1 3 0 4], rank {len(transpose)} must" +
        f" be equal to rank of input tensor, shape [1 3 2 7 5], rank {len(d1.shape)}."
    )


def test_asin(op_tester):
    # create test data
    d1 = ((np.random.rand(4) - 0.5) * np.pi).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.asin([i1])
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        a = torch.tensor(d1, requires_grad=True)
        out = torch.asin(a)
        return [out]

    op_tester.setPatterns(['PreUniRepl'], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'infer')


def test_asin_inplace(op_tester):
    # create test data
    d1 = ((np.random.rand(4) - 0.5) * np.pi).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.asin([i1])
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        a = torch.tensor(d1, requires_grad=True)
        out = torch.asin(a)
        return [out]

    op_tester.setPatterns(['InPlace'], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'infer')


def test_asin_grad(op_tester):
    # create the test data
    d1 = ((np.random.rand(4) - 0.5) * np.pi).astype(np.float32)
    print(d1)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.asin([i1])
        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + i1,
            popart.reservedGradientPrefix() + o,
        ]

    def reference(ref_data):
        a = torch.tensor(d1, requires_grad=True)
        out = torch.asin(a)
        d__o = ref_data.getOutputTensorGrad(0)
        out.backward(torch.tensor(d__o))
        return [out, a.grad, None]

    # op_tester.setPatterns(['OpToIdentity'], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'train')


def test_acos(op_tester):
    # create test data
    d1 = ((np.random.rand(4) - 0.5) * np.pi).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.acos([i1])
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        a = torch.tensor(d1, requires_grad=True)
        out = torch.acos(a)
        return [out]

    op_tester.setPatterns(['DecomposeBinaryConstScalar'],
                          enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'infer')


def test_acos_inplace(op_tester):
    # create test data
    d1 = ((np.random.rand(4) - 0.5) * np.pi).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.acos([i1])
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        a = torch.tensor(d1, requires_grad=True)
        out = torch.acos(a)
        return [out]

    op_tester.setPatterns(['InPlace', 'DecomposeBinaryConstScalar'],
                          enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'infer')


def test_acos_grad(op_tester):
    # create the test data
    d1 = ((np.random.rand(4) - 0.5) * np.pi).astype(np.float32)
    print(d1)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.acos([i1])
        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + i1,
            popart.reservedGradientPrefix() + o,
        ]

    def reference(ref_data):
        a = torch.tensor(d1, requires_grad=True)
        out = torch.acos(a)
        d__o = ref_data.getOutputTensorGrad(0)
        out.backward(torch.tensor(d__o))
        return [out, a.grad, None]

    op_tester.setPatterns(['DecomposeBinaryConstScalar', 'SubtractArg1GradOp'],
                          enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'train')


def test_acosh(op_tester):
    # create test data
    d1 = np.array([1.0, 1.2, 2.0, 3.0, 10.0, 100.0, 2001.0], dtype=np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.acosh([i1])
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        out = np.arccosh(d1)
        return [out]

    op_tester.setPatterns(['DecomposeBinaryConstScalar'],
                          enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'infer')


def test_acosh_inplace(op_tester):
    # create test data
    d1 = np.array([1.0, 1.2, 2.0, 3.0, 10.0, 100.0, 2001.0], dtype=np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.acosh([i1])
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        out = np.arccosh(d1)
        return [out]

    op_tester.setPatterns(['InPlace', 'DecomposeBinaryConstScalar'],
                          enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'infer')


def test_acosh_grad(op_tester):
    # create test data
    # This test fails for x "very" close to 1, e.g.  1.001.
    # Acosh is defined for x > 1.
    d1 = np.array([1.005, 1.2, 2.0, 3.0, 10.0, 10.123456, 100.0, 2001.0],
                  dtype=np.float32)

    def derivative_acosh(x):
        return 1 / (np.sqrt(x - 1) * np.sqrt(x + 1))

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.acosh([i1])
        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + i1,
            popart.reservedGradientPrefix() + o,
        ]

    def reference(ref_data):
        out = np.arccosh(d1)
        d__o = derivative_acosh(d1) * ref_data.getOutputTensorGrad(0)
        return [out, d__o, None]

    op_tester.setPatterns([
        'DecomposeBinaryConstScalar', 'SubtractArg1GradOp', 'LogGradOp',
        'SqrtGradOp', 'PowArg0GradOp'
    ],
                          enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'train')


def test_atan(op_tester):
    # create test data
    d1 = (np.random.rand(4)).astype(np.float32)

    print(d1)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.atan([i1])
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        a = torch.tensor(d1, requires_grad=True)
        out = torch.atan(a)
        return [out]

    # op_tester.setPatterns(['PreUniRepl'], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'infer')


def test_atan_inplace(op_tester):
    # create test data
    d1 = ((np.random.rand(4) - 0.5) * np.pi).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.atan([i1])
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        a = torch.tensor(d1, requires_grad=True)
        out = torch.atan(a)
        return [out]

    op_tester.setPatterns(['InPlace'], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'infer')


def test_atan_grad(op_tester):
    # create the test data
    d1 = ((np.random.rand(4) - 0.5) * np.pi).astype(np.float32)
    print(d1)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.atan([i1])
        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + i1,
            popart.reservedGradientPrefix() + o,
        ]

    def reference(ref_data):
        a = torch.tensor(d1, requires_grad=True)
        out = torch.atan(a)
        d__o = ref_data.getOutputTensorGrad(0)
        out.backward(torch.tensor(d__o))
        return [out, a.grad, None]

    op_tester.run(init_builder, reference, 'train')


def test_sinh(op_tester):
    # create test data
    d1 = np.random.rand(4).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.sinh([i1])
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        a = torch.tensor(d1, requires_grad=True)
        out = torch.sinh(a)
        return [out]

    op_tester.setPatterns(['PreUniRepl'], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'infer')


def test_sinh_inplace(op_tester):
    # create test data
    d1 = np.random.rand(4).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.sinh([i1])
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        a = torch.tensor(d1, requires_grad=True)
        out = torch.sinh(a)
        return [out]

    op_tester.setPatterns(['InPlace'], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'infer')


def test_sinh_grad(op_tester):
    # create the test data
    d1 = np.random.rand(4).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.sinh([i1])
        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + i1,
            popart.reservedGradientPrefix() + o,
        ]

    def reference(ref_data):
        a = torch.tensor(d1, requires_grad=True)
        out = torch.sinh(a)
        d__o = ref_data.getOutputTensorGrad(0)
        out.backward(torch.tensor(d__o))
        return [out, a.grad, None]

    op_tester.setPatterns(['OpToIdentity'], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'train')


def test_log(op_tester):
    # create test data
    d1 = np.random.rand(4).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.log([i1])
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        a = torch.tensor(d1, requires_grad=True)
        b = torch.log(a)
        return [b]

    op_tester.run(init_builder, reference, 'infer')


def test_log_grad(op_tester):
    # create test data
    d1 = np.random.rand(4).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.log([i1])
        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + i1,
            popart.reservedGradientPrefix() + o
        ]

    def reference(ref_data):
        a = torch.tensor(d1, requires_grad=True)
        b = torch.log(a)
        d__o = ref_data.getOutputTensorGrad(0)
        b.backward(torch.tensor(d__o))
        return [b, a.grad, None]

    op_tester.setPatterns(['PreUniRepl', 'LogGradOp'],
                          enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'train')


def test_unsqueeze(op_tester):
    d1 = np.random.rand(3, 4, 5).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.unsqueeze([i1], axes=[0, 4])
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        o = d1
        for i in (0, 4):
            o = np.expand_dims(o, axis=i)
        return [o]

    op_tester.setPatterns([], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'infer')


def test_unsqueeze_grad(op_tester):
    d1 = np.random.rand(3, 4, 5).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.unsqueeze([i1], axes=[0, 4])
        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + i1,
            popart.reservedGradientPrefix() + o
        ]

    def reference(ref_data):
        a = torch.tensor(d1, requires_grad=True)
        o = torch.unsqueeze(a, 0)
        o = torch.unsqueeze(o, 4)
        d__o = ref_data.getOutputTensorGrad(0)
        o.backward(torch.tensor(d__o))
        return [o, a.grad, None]

    op_tester.setPatterns(['PreUniRepl'], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'train')


def test_pad(op_tester):
    data = np.array([[[1., 2.], [3., 4.]]]).astype(np.float32)
    _test_pad(op_tester,
              data,
              lower_padding=(2, 1, 1),
              upper_padding=(1, 0, 2),
              mode='constant')


def test_pad_with_value(op_tester):
    data = np.array([[[1., 2.], [3., 4.]]]).astype(np.float32)
    _test_pad(op_tester,
              data,
              lower_padding=(2, 1, 1),
              upper_padding=(1, 0, 2),
              mode='constant',
              pad_value=0.3)


def test_pad_type_edge(op_tester):
    data = np.array([[[1., 2.], [3., 4.]]]).astype(np.float32)
    _test_pad(op_tester,
              data,
              lower_padding=(2, 1, 1),
              upper_padding=(1, 0, 2),
              mode='edge')


def test_pad_type_reflect(op_tester):
    data = np.array([[1., 2., 3.], [4., 5., 6.]]).astype(np.float32)
    _test_pad(op_tester,
              data,
              lower_padding=(1, 0),
              upper_padding=(0, 2),
              mode='reflect')


def _negative_padding(data, lower_padding, upper_padding):
    slices = []
    for idx, dim in enumerate(data.shape):
        # This only works for negative padding
        assert lower_padding[idx] <= 0
        assert upper_padding[idx] <= 0
        start = 0 - lower_padding[idx]
        stop = dim + upper_padding[idx]
        slices.append(slice(start, stop))
    return data[slices]


# Test that the local function `_negative_padding` works as expected.
def test_negative_padding_func():
    data = np.random.rand(2, 2, 4, 4).astype(np.float32)
    ref = data[:, :, 1:-1, 1:-2]
    x = _negative_padding(data, [0, 0, -1, -1], [0, 0, -1, -2])
    assert ref.shape == x.shape
    assert np.allclose(x, ref)


# Test popart `pad` supports negative padding.
def test_pad_negative_padding(op_tester):
    data = np.random.rand(2, 2, 4, 4).astype(np.float32)
    lower_padding = (0, 0, -1, -1)
    upper_padding = (0, 0, -1, -2)

    def init_builder(builder):
        i1 = builder.addInputTensor(data)
        o = builder.aiOnnx.pad([i1],
                               pads=(lower_padding + upper_padding),
                               mode='constant',
                               value=0.0)
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        result = _negative_padding(data, lower_padding, upper_padding)
        return [result]

    op_tester.setPatterns(['PreUniRepl'], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'infer')


def test_pad11(op_tester):
    data = np.array([[[1., 2.], [3., 4.]]]).astype(np.float32)
    pads = np.array([2, 1, 1, 1, 0, 2]).astype(np.int64)
    pad_value = 1.0
    value = np.array([pad_value]).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(data)
        p = builder.aiOnnx.constant(pads, False)
        v = builder.aiOnnx.constant(value, False)
        o = builder.aiOnnx.pad([i1, p, v], mode='constant')
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        padding = tuple(zip(pads, pads[len(pads) // 2:]))
        o = np.pad(data, padding, 'constant', constant_values=pad_value)
        print(o)

        return [o]

    op_tester.setPatterns(['PreUniRepl'], enableRuntimeAsserts=False)
    op_tester.run(init_builder,
                  reference,
                  'infer',
                  opsets={
                      "ai.onnx": 11,
                      "ai.graphcore": 1
                  })


def _test_pad(op_tester, data, lower_padding, upper_padding, mode,
              pad_value=0):
    def init_builder(builder):
        i1 = builder.addInputTensor(data)
        o = builder.aiOnnx.pad([i1],
                               pads=(lower_padding + upper_padding),
                               mode=mode,
                               value=pad_value)
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        padding = tuple(zip(lower_padding, upper_padding))
        if mode == 'constant':
            o = np.pad(data, padding, mode, constant_values=pad_value)
        else:
            o = np.pad(data, padding, mode)

        return [o]

    op_tester.setPatterns(['PreUniRepl'], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'infer')


def test_pad_grad(op_tester):
    d1 = np.array([[1., 2., 3., 4.], [5., 6., 7., 8.]]).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.pad([i1], [0, 2, 1, 1], "constant", 0)
        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + i1,
            popart.reservedGradientPrefix() + o
        ]

    def reference(ref_data):

        a = torch.tensor(d1, requires_grad=True)
        o = F.pad(input=a, pad=(2, 1, 0, 1), mode='constant', value=0)

        d__o = ref_data.getOutputTensorGrad(0)

        o.backward(torch.tensor(d__o))

        return [o, a.grad, d__o]

    op_tester.setPatterns(['PreUniRepl'], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'train')


def test_shape(op_tester):
    d1 = np.random.rand(2, 4, 3).astype(np.float32)
    d2 = np.zeros((4, 6), dtype=np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        c = builder.aiOnnx.shape([i2])
        o = builder.aiOnnx.reshape([i1, c])
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        out = np.reshape(d1, d2.shape)
        return [out]

    op_tester.run(init_builder, reference, 'infer')


def test_shape2(op_tester):
    d1 = np.random.rand(2, 4, 3).astype(np.float32)
    d2 = np.zeros((4, 6), dtype=np.float32)

    def init_builder(builder):
        i1 = builder.aiOnnx.constant(d1)
        i2 = builder.aiOnnx.constant(d2)
        c = builder.aiOnnx.shape([i2])
        o = builder.aiOnnx.reshape([i1, c])
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        out = np.reshape(d1, d2.shape)
        return [out]

    op_tester.run(init_builder, reference, 'infer')


def test_flatten_infer(op_tester):
    d1 = np.random.rand(2, 3, 4, 5).astype(np.float32)
    axis = 2

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.flatten([i1], axis, "test_flatten")
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        shape = d1.shape
        new_shape = (1,
                     -1) if axis == 0 else (np.prod(shape[0:axis]).astype(int),
                                            -1)
        out = np.reshape(d1, new_shape)
        return [out]

    op_tester.setPatterns([], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'infer')
