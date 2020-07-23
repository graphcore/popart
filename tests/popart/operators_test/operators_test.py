# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
import torch
import pytest
import torch.nn.functional as F
from op_tester import op_tester

# `import test_util` requires adding to sys.path
import sys
from pathlib import Path
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
        d__o = ref_data.getOutputTensorGrad(0)
        out.backward(torch.tensor(d__o))
        return [out, t1.grad, t2.grad, None]

    op_tester.setPatterns(['PreUniRepl'], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'train')


def test_cast(op_tester):
    d1 = np.random.uniform(0, 20, 5).astype(np.int32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.cast([i1], "FLOAT")

        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        return [d1.astype(np.float32)]

    op_tester.run(init_builder, reference, 'infer')


def test_cast_grad(op_tester):
    d1 = np.random.uniform(0, 10, 10).astype(np.int32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        c = builder.aiOnnx.cast([i1], "FLOAT")
        # Add an op that produces a gradient so we can test CastGrad properly
        o = builder.aiOnnx.sqrt([c])
        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + i1,
            popart.reservedGradientPrefix() + o
        ]

    def reference(ref_data):
        c = torch.tensor(d1, dtype=torch.float32, requires_grad=True)
        out = torch.sqrt(c)
        d_o = ref_data.getOutputTensorGrad(0)
        out.backward(torch.tensor(d_o))
        d_i1 = c.grad.numpy().astype(np.int32)
        return [out, d_i1, d_o]

    op_tester.setPatterns(['PreUniRepl', 'PostNRepl', 'SqrtGradOp'],
                          enableRuntimeAsserts=False)
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

        def reference(ref_data):
            if then_branch is True:
                return [np.asarray([7, 22, 4]).astype(np.int32)]
            else:
                return [np.asarray([-7, 18, 6]).astype(np.int32)]

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

    def reference(ref_data):
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

    def reference(ref_data):
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

    def reference(ref_data):
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

    def reference(ref_data):
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

    def reference(ref_data):
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

    def reference(ref_data):
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

    def reference(ref_data):
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

    def reference(ref_data):
        d = torch.tensor(data)
        conv = torch.nn.Conv1d(chans_in,
                               chans_out,
                               kernel_size,
                               padding=[padding])
        conv.weight.data = torch.tensor(filt)
        conv.bias.data = torch.tensor(bias)
        o = conv(d)
        return [o]

    op_tester.setPatterns(['SplitConvBias'], enableRuntimeAsserts=False)
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
    def reference(ref_data):
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

    def reference(ref_data):
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
        d__o = ref_data.getOutputTensorGrad(0)
        out.backward(torch.tensor(d__o))
        return [out, t1.grad, t2.grad, None]

    op_tester.setPatterns(['PreUniRepl', 'DivArg0GradOp', 'DivArg1GradOp'],
                          enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'train')


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


def test_sqrt(op_tester):
    # create test data
    d1 = np.random.rand(4).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.sqrt([i1])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
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


def test_exp(op_tester):
    # create test data
    d1 = np.random.rand(4).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.exp([i1])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
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

    def reference(ref_data):
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


def test_transpose(op_tester):
    d1 = np.random.rand(3, 5, 2, 7).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.transpose([i1], [2, 0, 3, 1])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
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

    def reference(ref_data):
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

    def reference(ref_data):
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

    def reference(ref_data):
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


def test_atan(op_tester):
    # create test data
    d1 = (np.random.rand(4)).astype(np.float32)

    print(d1)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.atan([i1])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
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

    def reference(ref_data):
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

    # op_tester.setPatterns(['OpToIdentity'], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'train')


def test_sinh(op_tester):
    # create test data
    d1 = np.random.rand(4).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.sinh([i1])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
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

    def reference(ref_data):
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

    def reference(ref_data):
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


def test_logsoftmax(op_tester):
    # create test data
    # Note: poplar implementation of softmax
    # requires outer 'batch' dimension
    d1 = np.random.rand(1, 4).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.logsoftmax([i1])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        a = torch.tensor(d1, requires_grad=True)
        # 'dim' corresponds to dim index over which
        # to perform softmax
        lsm = torch.nn.LogSoftmax(dim=1)
        b = lsm(a)
        return [b]

    op_tester.setPatterns(['LogSoftmaxOp', 'LogGradOp'],
                          enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'infer')


def test_logsoftmax_grad(op_tester):
    # create test data
    d1 = np.random.rand(1, 10).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.logsoftmax([i1])
        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + i1,
            popart.reservedGradientPrefix() + o
        ]

    def reference(ref_data):
        a = torch.tensor(d1, requires_grad=True)
        lsm = torch.nn.LogSoftmax(dim=1)
        b = lsm(a)
        d__o = ref_data.getOutputTensorGrad(0)
        b.backward(torch.tensor(d__o))
        return [b, a.grad, None]

    op_tester.atol *= 10
    op_tester.setPatterns(['PreUniRepl', 'LogSoftmaxOp', 'LogGradOp'],
                          enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'train')


def test_unsqueeze(op_tester):
    d1 = np.random.rand(3, 4, 5).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.unsqueeze([i1], axes=[0, 4])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        o = d1
        for i in (0, 4):
            o = np.expand_dims(o, axis=i)
        return [o]

    op_tester.setPatterns(['OpToReshape'], enableRuntimeAsserts=False)
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

    op_tester.setPatterns(['PreUniRepl', 'OpToReshape'],
                          enableRuntimeAsserts=False)
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

    def reference(ref_data):
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


def _test_pad(op_tester,
              data,
              lower_padding,
              upper_padding,
              mode,
              pad_value=0):
    def init_builder(builder):
        i1 = builder.addInputTensor(data)
        o = builder.aiOnnx.pad([i1],
                               pads=(lower_padding + upper_padding),
                               mode=mode,
                               value=pad_value)
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
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


def test_scatter_0(op_tester):
    data = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0]]).astype(np.float32)
    indices = np.array([[1, 0, 2], [0, 2, 1]]).astype(np.int32)
    updates = np.array([[-1.0, -1.1, -1.2], [2.0, 2.1,
                                             2.2]]).astype(np.float32)
    output = np.array([[2.0, -1.1, 0.0], [-1.0, 0.0, 2.2],
                       [0.0, 2.1, -1.2]]).astype(np.float32)
    axis = 0

    def init_builder(builder):
        i1 = builder.addInputTensor(data)
        i2 = builder.addInputTensor(indices)
        i3 = builder.addInputTensor(updates)
        o = builder.aiOnnx.scatter([i1, i2, i3], axis)
        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + i1,
            popart.reservedGradientPrefix() + i3
        ]

    def reference(ref_data):
        data_grad = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0],
                              [1.0, 0.0, 0.0]]).astype(np.float32)
        return [output, data_grad, np.ones_like(updates)]

    op_tester.lossReduction = popart.ReductionType.Sum
    op_tester.setPatterns(['PreUniRepl'], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'train')


def test_scatter_1(op_tester):
    data = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]]).astype(np.float32)
    indices = np.array([[1, 3]]).astype(np.int32)
    updates = np.array([[-1.1, 2.1]]).astype(np.float32)
    output = np.array([[1.0, -1.1, 3.0, 2.1, 5.0]]).astype(np.float32)
    d_data = np.array([[1.0, 0, 1.0, 0, 1.0]]).astype(np.float32)
    axis = 1

    def init_builder(builder):
        i1 = builder.addInputTensor(data)
        i2 = builder.addInputTensor(indices)
        i3 = builder.addInputTensor(updates)
        o = builder.aiOnnx.scatter([i1, i2, i3], axis)
        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + i1,
            popart.reservedGradientPrefix() + i3
        ]

    def reference(ref_data):
        return [output, d_data, np.ones_like(updates)]

    op_tester.lossReduction = popart.ReductionType.Sum
    op_tester.setPatterns(['PreUniRepl'], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'train')


def test_scatter_2(op_tester):
    data = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0]]).astype(np.float32)
    indices = np.array([[1, 0, 2], [0, 2, 1]]).astype(np.int32)
    updates = np.array([[-1.0, -1.1, -1.2], [2.0, 2.1,
                                             2.2]]).astype(np.float32)
    output = np.array([[-1.1, -1, -1.2], [2, 2.2, 2.1],
                       [0.0, 0.0, 0.0]]).astype(np.float32)
    axis = 1

    def init_builder(builder):
        i1 = builder.addInputTensor(data)
        i2 = builder.addInputTensor(indices)
        i3 = builder.addInputTensor(updates)
        o = builder.aiOnnx.scatter([i1, i2, i3], axis)
        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + i1,
            popart.reservedGradientPrefix() + i3
        ]

    def reference(ref_data):
        data_grad = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                              [1.0, 1.0, 1.0]]).astype(np.float32)
        return [output, data_grad, np.ones_like(updates)]

    op_tester.lossReduction = popart.ReductionType.Sum
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

    def reference(ref_data):
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

    def reference(ref_data):
        shape = d1.shape
        new_shape = (1,
                     -1) if axis == 0 else (np.prod(shape[0:axis]).astype(int),
                                            -1)
        out = np.reshape(d1, new_shape)
        return [out]

    op_tester.setPatterns(['OpToReshape'], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'infer')


def test_argmin_no_keepdims(op_tester):
    d1 = np.random.rand(5, 7, 11, 13).astype(np.float32)
    axis = 0
    keepdims = 0

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.argmin([i1], axis, keepdims, "test_argmin")
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        result = np.argmin(d1, axis=axis)
        return [result.astype(np.int32)]

    op_tester.run(init_builder, reference, 'infer')


def test_argmin_keepdims(op_tester):
    d1 = np.random.rand(5, 7, 11, 13).astype(np.float32)
    axis = 0
    keepdims = 1

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.argmin([i1], axis, keepdims, "test_argmin")
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        result = np.argmin(d1, axis=axis)
        result = np.expand_dims(result, axis)
        return [result.astype(np.int32)]

    op_tester.run(init_builder, reference, 'infer')


def test_argmin_negative_axis(op_tester):
    d1 = np.random.rand(5, 7, 11, 13).astype(np.float32)
    axis = -1
    keepdims = 1

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.argmin([i1], axis, keepdims, "test_argmin")
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        result = np.argmin(d1, axis=axis)
        result = np.expand_dims(result, axis)
        return [result.astype(np.int32)]

    op_tester.run(init_builder,
                  reference,
                  'infer',
                  opsets={
                      "ai.onnx": 11,
                      "ai.graphcore": 1
                  })


def _test_argmax(op_tester, data, axis, keepdims, opsets):
    def init_builder(builder):
        i1 = builder.addInputTensor(data)
        o = builder.aiOnnx.argmax([i1], axis, keepdims, "test_argmax")
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        result = np.argmax(data, axis=axis)
        if keepdims == 1:
            result = np.expand_dims(result, axis)
        return [result.astype(np.int32)]

    op_tester.run(init_builder, reference, 'infer', opsets=opsets)


def test_argmax_2d(op_tester):
    data = np.random.rand(5, 6).astype(np.float32)
    opsets = {"ai.onnx": 9, "ai.graphcore": 1}
    _test_argmax(op_tester, data, 0, 1, opsets)
    _test_argmax(op_tester, data, 0, 0, opsets)
    _test_argmax(op_tester, data, 1, 1, opsets)
    _test_argmax(op_tester, data, 1, 0, opsets)

    # Test negative axis index for onnx opset 11
    opsets = {"ai.onnx": 11, "ai.graphcore": 1}
    _test_argmax(op_tester, data, -1, 0, opsets)
    _test_argmax(op_tester, data, -2, 0, opsets)


def test_argmax_no_keepdims(op_tester):
    d1 = np.random.rand(5, 7, 11, 13).astype(np.float32)
    axis = 0
    keepdims = 0

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.argmax([i1], axis, keepdims, "test_argmax")
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        result = np.argmax(d1, axis=axis)
        return [result.astype(np.int32)]

    op_tester.run(init_builder, reference, 'infer')


def test_ceil(op_tester):
    d1 = np.random.rand(2, 7).astype(np.float32)
    d1 = d1 * 6 - 3  # numbers in range [-3, 3]

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.ceil([i1], "test_ceil")
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        result = np.ceil(d1)
        return [result.astype(np.float32)]

    op_tester.run(init_builder, reference, 'infer')


def test_ceil_inplace(op_tester):
    d1 = np.random.rand(2, 7).astype(np.float32)
    d1 = d1 * 10  # numbers in range [0, 10]

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        # Pad with ops to allow in-placing
        log = builder.aiOnnx.log([i1])
        ceil = builder.aiOnnx.ceil([log], "test_ceil")
        o = builder.aiOnnx.exp([ceil])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        result = np.exp(np.ceil(np.log(d1)))
        return [result.astype(np.float32)]

    op_tester.setPatterns(['InPlace'], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'infer')


def test_ceil_grad(op_tester):
    d1 = np.random.rand(2, 7).astype(np.float32)
    d1 = d1 * 6 - 3  # numbers in range [-3, 3]

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.ceil([i1], "test_ceil")
        builder.addOutputTensor(o)
        return [o, popart.reservedGradientPrefix() + o]

    def reference(ref_data):
        return [np.ceil(d1 * 0).astype(np.float32)]

    with pytest.raises(popart.popart_exception) as e_info:
        op_tester.run(init_builder, reference, 'train')

    assert (e_info.value.args[0].startswith(
        "PopART does not have a valid grad op"))


def test_floor(op_tester):
    d1 = np.random.rand(2, 7).astype(np.float32)
    d1 = d1 * 6 - 3  # numbers in range [-3, 3]

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.floor([i1], "test_floor")
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        result = np.floor(d1)
        return [result.astype(np.float32)]

    op_tester.run(init_builder, reference, 'infer')


def test_floor_inplace(op_tester):
    d1 = np.random.rand(2, 7).astype(np.float32)
    d1 = d1 * 10  # numbers in range [0, 10]

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        # Pad with ops to allow in-placing
        log = builder.aiOnnx.log([i1])
        floor = builder.aiOnnx.floor([log], "test_floor")
        o = builder.aiOnnx.exp([floor])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        result = np.exp(np.floor(np.log(d1)))
        return [result.astype(np.float32)]

    op_tester.setPatterns(['InPlace'], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'infer')


def test_floor_grad(op_tester):
    d1 = np.random.rand(2, 7).astype(np.float32)
    d1 = d1 * 6 - 3  # numbers in range [-3, 3]

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.floor([i1], "test_floor")
        builder.addOutputTensor(o)
        return [o, popart.reservedGradientPrefix() + o]

    def reference(ref_data):
        return [np.floor(d1 * 0).astype(np.float32)]

    with pytest.raises(popart.popart_exception) as e_info:
        op_tester.run(init_builder, reference, 'train')

    assert (e_info.value.args[0].startswith(
        "PopART does not have a valid grad op"))


def test_clip(op_tester):
    d1 = np.random.rand(2, 7).astype(np.float32)
    d1 = d1 * 6 - 3  # numbers in range [-3, 3]

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.clip([i1], min=-1.5, max=1.5)
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        a = torch.tensor(d1)
        result = torch.clamp(a, min=-1.5, max=1.5)
        return [result]

    op_tester.run(init_builder, reference, 'infer')


def test_clip_inplace(op_tester):
    d1 = np.random.rand(2, 7).astype(np.float32)
    d1 = d1 * 10  # numbers in range [0, 10]

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        # Pad with ops to allow in-placing
        log = builder.aiOnnx.log([i1])
        clip = builder.aiOnnx.clip([log], min=4, max=7)
        o = builder.aiOnnx.exp([clip])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        a = torch.tensor(d1)
        result = torch.exp(torch.clamp(torch.log(a), min=4, max=7))
        return [result]

    op_tester.setPatterns(['InPlace'], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'infer')


def test_clip_grad(op_tester):
    d1 = np.random.rand(2, 7).astype(np.float32)
    d1 = d1 * 6 - 3  # numbers in range [-3, 3]

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.clip([i1], min=-1.5, max=1.5)
        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + i1,
            popart.reservedGradientPrefix() + o
        ]

    def reference(ref_data):
        a = torch.tensor(d1, requires_grad=True)
        b = torch.clamp(a, min=-1.5, max=1.5)
        d__o = ref_data.getOutputTensorGrad(0)
        b.backward(torch.tensor(d__o))
        print(b)
        print(a.grad)
        print("b grad", b.grad)
        return [b, a.grad, None]

    op_tester.setPatterns(['PreUniRepl'], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'train')


def test_argmax_keepdims(op_tester):
    d1 = np.random.rand(5, 7, 11, 13).astype(np.float32)
    axis = 0
    keepdims = 1

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.argmax([i1], axis, keepdims, "test_argmax")
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        result = np.argmax(d1, axis=axis)
        result = np.expand_dims(result, axis)
        return [result.astype(np.int32)]

    op_tester.run(init_builder, reference, 'infer')


def test_instancenorm_grad(op_tester):
    batch_size = 3
    features = 3
    width = 4
    height = 4

    non_zero_places = 5

    data = np.random.rand(batch_size, features, width,
                          height).astype(np.float32)

    a_few_places = np.zeros_like(data.flatten())

    for _ in range(non_zero_places):
        a_few_places[int(np.random.rand() * a_few_places.size)] = 1.0

    a_few_places = a_few_places.reshape(*data.shape)

    scale = np.random.rand(features).astype(np.float32)
    bias = np.random.rand(features).astype(np.float32)

    epsilon = 1e-05

    def init_builder(builder):

        i_data = builder.addInputTensor(data)
        i_scale = builder.addInputTensor(scale)
        i_bias = builder.addInputTensor(bias)

        i_few_places = builder.addInputTensor(a_few_places)

        normed = builder.aiOnnx.instancenormalization(
            [i_data, i_scale, i_bias], epsilon)
        out = builder.aiOnnx.mul([normed, i_few_places])

        builder.addOutputTensor(out)

        return [
            out, normed,
            popart.reservedGradientPrefix() + i_data,
            popart.reservedGradientPrefix() + i_scale,
            popart.reservedGradientPrefix() + i_bias,
            popart.reservedGradientPrefix() + out
        ]

    def reference(ref_data):
        i_data = torch.tensor(data, requires_grad=True)

        m = torch.nn.InstanceNorm2d(features,
                                    eps=epsilon,
                                    momentum=0,
                                    affine=True)
        m.weight.data = torch.tensor(scale)
        m.bias.data = torch.tensor(bias)
        normed = m(i_data)

        out = normed * torch.tensor(a_few_places, requires_grad=False)

        d__o = ref_data.getOutputTensorGrad(0)

        out.backward(torch.tensor(d__o))

        assert i_data.grad is not None
        assert m.weight.grad is not None
        assert m.bias.grad is not None

        return [out, normed, i_data.grad, m.weight.grad, m.bias.grad, None]

    op_tester.atol *= 10
    op_tester.setPatterns(['PreUniRepl', 'ReciprocalGradOp', 'MulArgGradOp'],
                          enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'train')


def test_constantofshape(op_tester):
    shape = np.array([1, 2, 3]).astype(np.int64)
    value = np.array([3.1415]).astype(np.float32)

    def init_builder(builder):
        i = builder.aiOnnx.constant(shape)
        c = builder.aiOnnx.constantofshape([i], value)
        o = builder.aiOnnx.identity([c])

        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        out = np.array([3.1415] * 2 * 3).astype(np.float32)
        out = np.reshape(out, (1, 2, 3))
        return [out]

    op_tester.run(init_builder, reference, 'infer')
