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

        d__o = torch.tensor(ref_data.getOutputTensorGrad(0))
        assert not torch.isnan(d__o).any()
        out.backward(d__o)

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
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
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
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
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

    def reference(ref_data):
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

    def reference(ref_data):
        out = torch.flip(torch.tensor(d1), reverse_dims)
        return [out]

    op_tester.run(init_builder, reference, 'infer')

    with pytest.raises(popart.popart_exception) as e_info:
        op_tester.run(init_builder_negative_dim, reference, 'infer')
    assert "invalid dimension '-2'. Only positive dimensions are supported" in e_info.value.args[0]

    with pytest.raises(popart.popart_exception) as e_info:
        op_tester.run(init_builder_dim_appears_twice, reference, 'infer')
    assert "Dimension 1 appears multiple times" in e_info.value.args[0]

    with pytest.raises(popart.popart_exception) as e_info:
        op_tester.run(init_builder_dim_greater_than_rank, reference, 'infer')
    assert "invalid dimension '3' for input tensor of rank 3" in e_info.value.args[0]


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


def test_subtract(op_tester):
    d1 = np.random.rand(4).astype(np.float32)
    d2 = np.random.rand(4).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        o = builder.aiOnnx.sub([i1, i2], "test_subtract")
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
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

        def reference(ref_data):
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

        def reference(ref_data):
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

        sess = popart.InferenceSession(bld.getModelProto(),
                                       deviceInfo=tu.create_test_device(),
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


def test_acos(op_tester):
    # create test data
    d1 = ((np.random.rand(4) - 0.5) * np.pi).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.acos([i1])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
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

    def reference(ref_data):
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

    def reference(ref_data):
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

    def reference(ref_data):
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

    op_tester.setPatterns([], enableRuntimeAsserts=False)
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
        return [o, popart.reservedGradientPrefix() + i1]

    def reference(ref_data):
        return [np.ceil(d1).astype(np.float32), np.zeros_like(d1)]

    op_tester.run(init_builder, reference, 'train')


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
        return [o, popart.reservedGradientPrefix() + i1]

    def reference(ref_data):
        return [np.floor(d1).astype(np.float32), np.zeros_like(d1)]

    op_tester.run(init_builder, reference, 'train')


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


def test_clip11(op_tester):
    d1 = np.random.rand(2, 7).astype(np.float32)
    d1 = d1 * 6 - 3  # numbers in range [-3, 3]
    d_min = np.array([-1.5], dtype=np.float32)
    d_max = np.array([1.5], dtype=np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        t_min = builder.aiOnnx.constant(d_min, False)
        t_max = builder.aiOnnx.constant(d_max, False)
        o = builder.aiOnnx.clip([i1, t_min, t_max])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        a = torch.tensor(d1)
        result = torch.clamp(a, min=d_min[0], max=d_max[0])
        return [result]

    op_tester.run(init_builder,
                  reference,
                  'infer',
                  opsets={
                      "ai.onnx": 11,
                      "ai.graphcore": 1
                  })


def test_clip11_default_min(op_tester):
    d1 = np.random.rand(2, 7).astype(np.float32)
    d1 = d1 * 6 - 3  # numbers in range [-3, 3]
    d_max = np.array([1.5], dtype=np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        t_max = builder.aiOnnx.constant(d_max, False)
        o = builder.aiOnnx.clip([i1, '', t_max])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        a = torch.tensor(d1)
        result = torch.clamp(a,
                             min=torch.finfo(torch.float32).min,
                             max=d_max[0])
        return [result]

    op_tester.run(init_builder,
                  reference,
                  'infer',
                  opsets={
                      "ai.onnx": 11,
                      "ai.graphcore": 1
                  })


def test_clip11_default_max(op_tester):
    d1 = np.random.rand(2, 7).astype(np.float32)
    d1 = d1 * 6 - 3  # numbers in range [-3, 3]
    d_min = np.array([-1.5], dtype=np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        t_min = builder.aiOnnx.constant(d_min, False)
        o = builder.aiOnnx.clip([i1, t_min])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        a = torch.tensor(d1)
        # result = torch.clamp(a, min=d_min[0], max=torch.finfo(torch.float32).max)
        result = torch.clamp(a, min=d_min[0], max=100)
        return [result]

    op_tester.run(init_builder,
                  reference,
                  'infer',
                  opsets={
                      "ai.onnx": 11,
                      "ai.graphcore": 1
                  })


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


def test_instancenorm_grad_5D_input(op_tester):
    batch_size = 3
    features = 3
    d1 = 4
    d2 = 4
    d3 = 4

    non_zero_places = 10

    data = np.random.rand(batch_size, features, d1,
                          d2, d3).astype(np.float32)

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

        m = torch.nn.InstanceNorm3d(features,
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


def test_concat(op_tester):
    values = [np.random.rand(1, 2, 3).astype(np.float32) for i in range(4)]

    def init_builder(builder):
        i = [builder.addInputTensor(v) for v in values]
        c = builder.aiOnnx.concat(i, 1)
        o = builder.aiOnnx.identity([c])

        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        out = np.concatenate(values, 1)
        return [out]

    op_tester.run(init_builder, reference, 'infer')


def test_concat_negative_axis(op_tester):
    values = [np.random.rand(1, 2, 2, 2).astype(np.float32) for i in range(4)]

    def init_builder(builder):
        i = [builder.addInputTensor(v) for v in values]
        c = builder.aiOnnx.concat(i, -1)
        o = builder.aiOnnx.identity([c])

        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        out = np.concatenate(values, -1)
        return [out]

    op_tester.run(init_builder, reference, 'infer')


def test_constant_of_shape(op_tester):
    data = np.random.rand(2, 2).astype(np.float32)
    shape_data = np.array([2, 2]).astype(np.int64)

    def init_builder(builder):
        in0 = builder.addInputTensor(data)
        c0 = builder.aiOnnx.constant(shape_data)
        c = builder.aiOnnx.constantofshape([c0], np.array([5],
                                                          dtype=np.float32))
        o = builder.aiOnnx.add([in0, c])

        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        o = data + np.array([5], dtype=np.float32)
        return [o]

    op_tester.run(init_builder, reference, 'infer')


def test_constant_of_shape_int32(op_tester):
    data = np.random.rand(2, 2).astype(np.float32)
    shape_data = np.array([2, 2]).astype(np.int32)

    def init_builder(builder):
        in0 = builder.addInputTensor(data)
        c0 = builder.aiOnnx.constant(shape_data)
        c = builder.aiOnnx.constantofshape([c0], np.array([5],
                                                          dtype=np.float32))
        o = builder.aiOnnx.add([in0, c])

        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        o = data + np.array([5], dtype=np.float32)
        return [o]

    op_tester.run(init_builder, reference, 'infer')


def test_convtranspose(op_tester):
    # Test modified from example `convtranspose` in onnx
    # operators documentation.
    x = np.array([[[
        [0., 1., 2.],  # (1, 1, 3, 3)
        [3., 4., 5.],
        [6., 7., 8.]
    ]]]).astype(np.float32)

    W = np.array([[
        [
            [1., 1., 1.],  # (1, 2, 3, 3)
            [1., 1., 1.],
            [1., 1., 1.]
        ],
        [[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]]
    ]]).astype(np.float32)

    def init_builder(builder):
        d = builder.addInputTensor(x)
        f = builder.addInputTensor(W)
        o = builder.aiOnnxOpset11.convtranspose([d, f])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        y = np.array([[
            [
                [0., 1., 3., 3., 2.],  # (1, 2, 5, 5)
                [3., 8., 15., 12., 7.],
                [9., 21., 36., 27., 15.],
                [9., 20., 33., 24., 13.],
                [6., 13., 21., 15., 8.]
            ],
            [[0., 1., 3., 3., 2.], [3., 8., 15., 12., 7.],
             [9., 21., 36., 27., 15.], [9., 20., 33., 24., 13.],
             [6., 13., 21., 15., 8.]]
        ]]).astype(np.float32)
        return [y]

    op_tester.setPatterns(['ConvTranspose'], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, step_type='infer')


def test_convtranspose_1d(op_tester):
    # Test modified from example `convtranspose_1d` in onnx
    # operators documentation.
    x = np.array([[[0., 1., 2.]]]).astype(np.float32)  # (1, 1, 3)

    W = np.array([[
        [1., 1., 1.],  # (1, 2, 3)
        [1., 1., 1.]
    ]]).astype(np.float32)

    def init_builder(builder):
        d = builder.addInputTensor(x)
        f = builder.addInputTensor(W)
        o = builder.aiOnnxOpset11.convtranspose([d, f])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        y = np.array([[
            [0., 1., 3., 3., 2.],  # (1, 2, 5)
            [0., 1., 3., 3., 2.]
        ]]).astype(np.float32)
        return [y]

    op_tester.setPatterns(['ConvTranspose'], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, step_type='infer')


def test_convtranspose_3d(op_tester):
    # Test modified from example `convtranspose_3d` in onnx
    # operators documentation.
    x = np.array([[[[[0., 1., 2., 3., 4.], [5., 6., 7., 8., 9.],
                     [10., 11., 12., 13., 14.], [15., 16., 17., 18., 19.]],
                    [[20., 21., 22., 23., 24.], [25., 26., 27., 28., 29.],
                     [30., 31., 32., 33., 34.], [35., 36., 37., 38., 39.]],
                    [[40., 41., 42., 43., 44.], [45., 46., 47., 48., 49.],
                     [50., 51., 52., 53., 54.], [55., 56., 57., 58.,
                                                 59.]]]]]).astype(np.float32)

    W = np.array([[[[[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]],
                    [[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]],
                    [[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]]],
                   [[[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]],
                    [[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]],
                    [[1., 1., 1.], [1., 1., 1.], [1., 1.,
                                                  1.]]]]]).astype(np.float32)

    def init_builder(builder):
        d = builder.addInputTensor(x)
        f = builder.addInputTensor(W)
        o = builder.aiOnnxOpset11.convtranspose([d, f])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        y = np.array([[[[[0., 1., 3., 6., 9., 7., 4.],
                         [5., 12., 21., 27., 33., 24., 13.],
                         [15., 33., 54., 63., 72., 51., 27.],
                         [30., 63., 99., 108., 117., 81., 42.],
                         [25., 52., 81., 87., 93., 64., 33.],
                         [15., 31., 48., 51., 54., 37., 19.]],
                        [[20., 42., 66., 72., 78., 54., 28.],
                         [50., 104., 162., 174., 186., 128., 66.],
                         [90., 186., 288., 306., 324., 222., 114.],
                         [120., 246., 378., 396., 414., 282., 144.],
                         [90., 184., 282., 294., 306., 208., 106.],
                         [50., 102., 156., 162., 168., 114., 58.]],
                        [[60., 123., 189., 198., 207., 141., 72.],
                         [135., 276., 423., 441., 459., 312., 159.],
                         [225., 459., 702., 729., 756., 513., 261.],
                         [270., 549., 837., 864., 891., 603., 306.],
                         [195., 396., 603., 621., 639., 432., 219.],
                         [105., 213., 324., 333., 342., 231., 117.]],
                        [[60., 122., 186., 192., 198., 134., 68.],
                         [130., 264., 402., 414., 426., 288., 146.],
                         [210., 426., 648., 666., 684., 462., 234.],
                         [240., 486., 738., 756., 774., 522., 264.],
                         [170., 344., 522., 534., 546., 368., 186.],
                         [90., 182., 276., 282., 288., 194., 98.]],
                        [[40., 81., 123., 126., 129., 87., 44.],
                         [85., 172., 261., 267., 273., 184., 93.],
                         [135., 273., 414., 423., 432., 291., 147.],
                         [150., 303., 459., 468., 477., 321., 162.],
                         [105., 212., 321., 327., 333., 224., 113.],
                         [55., 111., 168., 171., 174., 117., 59.]]],
                       [[[0., 1., 3., 6., 9., 7., 4.],
                         [5., 12., 21., 27., 33., 24., 13.],
                         [15., 33., 54., 63., 72., 51., 27.],
                         [30., 63., 99., 108., 117., 81., 42.],
                         [25., 52., 81., 87., 93., 64., 33.],
                         [15., 31., 48., 51., 54., 37., 19.]],
                        [[20., 42., 66., 72., 78., 54., 28.],
                         [50., 104., 162., 174., 186., 128., 66.],
                         [90., 186., 288., 306., 324., 222., 114.],
                         [120., 246., 378., 396., 414., 282., 144.],
                         [90., 184., 282., 294., 306., 208., 106.],
                         [50., 102., 156., 162., 168., 114., 58.]],
                        [[60., 123., 189., 198., 207., 141., 72.],
                         [135., 276., 423., 441., 459., 312., 159.],
                         [225., 459., 702., 729., 756., 513., 261.],
                         [270., 549., 837., 864., 891., 603., 306.],
                         [195., 396., 603., 621., 639., 432., 219.],
                         [105., 213., 324., 333., 342., 231., 117.]],
                        [[60., 122., 186., 192., 198., 134., 68.],
                         [130., 264., 402., 414., 426., 288., 146.],
                         [210., 426., 648., 666., 684., 462., 234.],
                         [240., 486., 738., 756., 774., 522., 264.],
                         [170., 344., 522., 534., 546., 368., 186.],
                         [90., 182., 276., 282., 288., 194., 98.]],
                        [[40., 81., 123., 126., 129., 87., 44.],
                         [85., 172., 261., 267., 273., 184., 93.],
                         [135., 273., 414., 423., 432., 291., 147.],
                         [150., 303., 459., 468., 477., 321., 162.],
                         [105., 212., 321., 327., 333., 224., 113.],
                         [55., 111., 168., 171., 174., 117.,
                          59.]]]]]).astype(np.float32)

        return [y]

    op_tester.setPatterns(['ConvTranspose'], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, step_type='infer')


def test_convtranspose_pytorch(op_tester):
    def run_test(in_chans, out_chans, data, kernel):
        print(f'run_test({in_chans}, {out_chans}, {data}, {kernel})')
        assert len(data) == len(kernel)
        x = np.random.rand(1, in_chans, *data).astype(np.float32)
        W = np.random.rand(in_chans, out_chans, *kernel).astype(np.float32)

        def init_builder(builder):
            d = builder.addInputTensor(x)
            f = builder.addInputTensor(W)
            o = builder.aiOnnxOpset11.convtranspose([d, f])
            builder.addOutputTensor(o)
            return [o]

        def reference(ref_data):
            data = torch.tensor(x)
            weights = torch.tensor(W)
            if len(kernel) == 1:
                conv = torch.nn.ConvTranspose1d(in_chans, out_chans, kernel)
            elif len(kernel) == 2:
                conv = torch.nn.ConvTranspose2d(in_chans, out_chans, kernel)
            else:
                raise SystemError(f'Bad kernel size {len(kernel)}')
            conv.weight.data = weights
            conv.bias.data = torch.zeros(conv.bias.size())
            o = conv(data)
            print(o.shape)
            return [o]

        op_tester.setPatterns(['ConvTranspose'], enableRuntimeAsserts=False)
        op_tester.run(init_builder, reference, step_type='infer')

    # Test various 2d convtransposes
    run_test(in_chans=1, out_chans=2, data=[3, 3], kernel=[3, 3])
    run_test(in_chans=1, out_chans=2, data=[4, 4], kernel=[4, 4])
    run_test(in_chans=1, out_chans=2, data=[5, 5], kernel=[5, 5])
    run_test(in_chans=1, out_chans=2, data=[4, 4], kernel=[3, 3])
    run_test(in_chans=1, out_chans=2, data=[5, 5], kernel=[3, 3])
    run_test(in_chans=1, out_chans=2, data=[5, 5], kernel=[4, 4])
    run_test(in_chans=2, out_chans=3, data=[3, 3], kernel=[3, 3])
    run_test(in_chans=3, out_chans=6, data=[4, 4], kernel=[4, 4])

    # Test various 1d convtransposes
    run_test(in_chans=1, out_chans=2, data=[3], kernel=[3])
    run_test(in_chans=1, out_chans=2, data=[4], kernel=[4])
    run_test(in_chans=1, out_chans=2, data=[5], kernel=[5])
    run_test(in_chans=1, out_chans=2, data=[4], kernel=[3])
    run_test(in_chans=1, out_chans=2, data=[5], kernel=[3])
    run_test(in_chans=1, out_chans=2, data=[5], kernel=[4])
    run_test(in_chans=2, out_chans=3, data=[3], kernel=[3])
    run_test(in_chans=3, out_chans=6, data=[4], kernel=[4])


def test_convtranspose_pytorch_attributes(op_tester):
    def run_test(in_chans,
                 out_chans,
                 data,
                 kernel,
                 groups=1,
                 outshape=False,
                 stride=None,
                 output_padding=None,
                 pads=None):
        print(f'run_test({in_chans}, {out_chans}, {data}, {kernel})')
        assert len(data) == len(kernel)
        x = np.random.rand(1, in_chans, *data).astype(np.float32)
        W = np.random.rand(in_chans, out_chans // groups,
                           *kernel).astype(np.float32)
        bias = np.random.rand(out_chans).astype(np.float32)

        def reference(ref_data):
            data = torch.tensor(x)
            weights = torch.tensor(W)

            kwargs = {}
            if stride:
                kwargs['stride'] = stride
            if output_padding:
                kwargs['output_padding'] = output_padding
            if pads:
                kwargs['padding'] = pads
            kwargs['groups'] = groups

            if len(kernel) == 1:
                conv = torch.nn.ConvTranspose1d(in_chans, out_chans, kernel,
                                                **kwargs)
            elif len(kernel) == 2:
                conv = torch.nn.ConvTranspose2d(in_chans, out_chans, kernel,
                                                **kwargs)
            else:
                raise SystemError(f'Bad kernel size {len(kernel)}')
            conv.weight.data = weights
            conv.bias.data = torch.tensor(bias)
            o = conv(data)
            print(o.shape)
            return [o]

        torch_out_shape = None
        if outshape:
            torch_out_shape = reference(x)[0].shape

        def init_builder(builder):
            d = builder.addInputTensor(x)
            f = builder.addInputTensor(W)
            b = builder.addInitializedInputTensor(bias)

            kwargs = {}
            if stride:
                kwargs['strides'] = stride
            if output_padding:
                kwargs['output_padding'] = output_padding
            if pads:
                kwargs['pads'] = pads + pads

            if torch_out_shape:
                kwargs['output_shape'] = torch_out_shape

            kwargs['group'] = groups
            o = builder.aiOnnxOpset10.convtranspose([d, f, b], **kwargs)
            builder.addOutputTensor(o)
            return [o]

        op_tester.setPatterns(popart.Patterns(popart.PatternsLevel.Default))
        op_tester.run(init_builder, reference, step_type='infer')

    # just testing strides
    run_test(in_chans=1, out_chans=2, data=[4], kernel=[4], stride=[5])
    run_test(in_chans=1, out_chans=2, data=[5], kernel=[4], stride=[5])
    run_test(in_chans=2, out_chans=3, data=[5], kernel=[4], stride=[5])
    run_test(in_chans=2,
             out_chans=3,
             data=[5, 5],
             kernel=[4, 4],
             stride=[3, 5])

    # testing output padding
    run_test(in_chans=1,
             out_chans=2,
             data=[4],
             kernel=[4],
             stride=[2],
             output_padding=[1])

    # testing pads
    run_test(in_chans=1, out_chans=2, data=[3], kernel=[3], pads=[1])
    run_test(in_chans=1, out_chans=2, data=[5], kernel=[3], pads=[1])
    run_test(in_chans=1, out_chans=2, data=[5], kernel=[5], pads=[2])
    run_test(in_chans=1, out_chans=2, data=[5], kernel=[7], pads=[2])

    # stride and pads
    run_test(in_chans=1,
             out_chans=2,
             data=[3, 3],
             kernel=[3, 3],
             stride=[3, 2],
             pads=[1, 2],
             output_padding=[2, 1])

    # Test groups
    run_test(in_chans=4, out_chans=4, data=[3], kernel=[3], pads=[1], groups=4)
    run_test(in_chans=8, out_chans=8, data=[5], kernel=[3], pads=[1], groups=2)

    # Test output shape
    run_test(in_chans=8,
             out_chans=8,
             data=[5],
             kernel=[3],
             pads=[1],
             outshape=True)


def test_convtranspose_debug(op_tester):
    x = np.array([[[
        [0., 1., 2.],  # (1, 1, 3, 3)
        [3., 4., 5.],
        [6., 7., 8.]
    ]]]).astype(np.float32)

    W = np.array([[
        [
            [1., 1., 1.],  # (1, 2, 3, 3)
            [1., 1., 1.],
            [1., 1., 1.]
        ],
        [[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]]
    ]]).astype(np.float32)

    y = np.random.rand(1, 2, 10, 8).astype(np.float32)

    # y = np.array([[
    #     [
    #         [0., 1., 3., 3., 2.],  # (1, 2, 5, 5)
    #         [3., 8., 15., 12., 7.],
    #         [9., 21., 36., 27., 15.],
    #         [9., 20., 33., 24., 13.],
    #         [6., 13., 21., 15., 8.]
    #     ],
    #     [[0., 1., 3., 3., 2.], [3., 8., 15., 12., 7.],
    #      [9., 21., 36., 27., 15.], [9., 20., 33., 24., 13.],
    #      [6., 13., 21., 15., 8.]]
    # ]]).astype(np.float32)

    def init_builder(builder):
        d = builder.addInputTensor(y)
        f = builder.addInputTensor(W)
        o = builder.aiOnnxOpset11.conv([d, f],
                                       dilations=[3, 2],
                                       pads=[0, 0, -1, -1])
        builder.addOutputTensor(o)
        # return [o]
        return [o, popart.reservedGradientPrefix() + d]

    def reference(ref_data):
        return [None, None]

    op_tester.setPatterns(['ConvDataGrad'], enableRuntimeAsserts=False)
    # op_tester.run(init_builder, reference, step_type='infer')
    op_tester.run(init_builder, reference, step_type='train')


def test_where_0(op_tester):
    condition = np.array([[1, 0], [1, 1]], dtype=np.bool)
    x = np.array([[1, 2], [3, 4]], dtype=np.float32)
    y = np.array([[9, 8], [7, 6]], dtype=np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(condition)
        i2 = builder.addInputTensor(x)
        i3 = builder.addInputTensor(y)
        o = builder.aiOnnx.where([i1, i2, i3])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        ct = torch.tensor(condition)
        cx = torch.tensor(x)
        cy = torch.tensor(y)
        out = torch.where(ct, cx, cy)
        return [out]

    op_tester.run(init_builder, reference, 'infer')


def test_round(op_tester):
    d1 = np.random.rand(2, 7).astype(np.float32)
    d1 = d1 * 6 - 3  # numbers in range [-3, 3]

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnxOpset11.round([i1], "test_round")
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        result = np.round(d1)
        return [result.astype(np.float32)]

    op_tester.run(init_builder, reference, 'infer')


def test_round_graphcore(op_tester):
    d1 = np.random.rand(2, 7).astype(np.float32)
    d1 = d1 * 6 - 3  # numbers in range [-3, 3]

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiGraphcore.round([i1], "test_round")
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        result = np.round(d1)
        return [result.astype(np.float32)]

    op_tester.run(init_builder, reference, 'infer')


def test_round_inplace(op_tester):
    d1 = np.random.rand(2, 7).astype(np.float32)
    d1 = d1 * 10  # numbers in range [0, 10]

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        # Pad with ops to allow in-placing
        log = builder.aiOnnxOpset11.log([i1])
        round = builder.aiOnnxOpset11.round([log], "test_round")
        o = builder.aiOnnxOpset11.exp([round])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        result = np.exp(np.round(np.log(d1)))
        return [result.astype(np.float32)]

    op_tester.setPatterns(['InPlace'], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, 'train')


def test_round_grad(op_tester):
    d1 = np.random.rand(2, 7).astype(np.float32)
    d1 = d1 * 6 - 3  # numbers in range [-3, 3]

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnxOpset11.round([i1], "test_round")
        builder.addOutputTensor(o)
        return [o, popart.reservedGradientPrefix() + i1]

    def reference(ref_data):
        return [np.round(d1).astype(np.float32), np.zeros_like(d1)]

    op_tester.run(init_builder, reference, 'train')


def test_where_1(op_tester):
    condition = np.array([[1, 0], [1, 1]], dtype=np.bool)
    x = np.array([[1, 2]], dtype=np.float32)
    y = np.array([[9, 8], [7, 6]], dtype=np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(condition)
        i2 = builder.addInputTensor(x)
        i3 = builder.addInputTensor(y)
        o = builder.aiOnnx.where([i1, i2, i3])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        ct = torch.tensor(condition)
        cx = torch.tensor(x)
        cy = torch.tensor(y)
        out = torch.where(ct, cx, cy)
        return [out]

    op_tester.run(init_builder, reference, 'infer')


def test_where_2(op_tester):
    x = np.arange(9, dtype=np.int32)
    y = 10 * x
    condition = x < 5

    def init_builder(builder):
        i1 = builder.addInputTensor(condition)
        i2 = builder.addInputTensor(x)
        i3 = builder.addInputTensor(y)
        o = builder.aiOnnx.where([i1, i2, i3])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        ct = torch.tensor(condition)
        cx = torch.tensor(x)
        cy = torch.tensor(y)
        out = torch.where(ct, cx, cy)
        return [out]

    op_tester.run(init_builder, reference, 'infer')


def test_where_3(op_tester):
    x, y = np.ogrid[:3, :4]
    x = np.float32(x)
    y = np.float32(y)
    y = 10 + y
    condition = x < y

    def init_builder(builder):
        i1 = builder.addInputTensor(condition)
        i2 = builder.addInputTensor(x)
        i3 = builder.addInputTensor(y)
        o = builder.aiOnnx.where([i1, i2, i3])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        ct = torch.tensor(condition)
        cx = torch.tensor(x)
        cy = torch.tensor(y)
        out = torch.where(ct, cx, cy)
        return [out]

    op_tester.run(init_builder, reference, 'infer')


def test_where_4(op_tester):
    x = np.array([[0, 1, 2], [0, 2, 4], [0, 3, 6]])
    y = np.array([-1])
    x = np.float32(x)
    y = np.float32(y)
    condition = x < 4

    def init_builder(builder):
        i1 = builder.addInputTensor(condition)
        i2 = builder.addInputTensor(x)
        i3 = builder.addInputTensor(y)
        o = builder.aiOnnx.where([i1, i2, i3])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        ct = torch.tensor(condition)
        cx = torch.tensor(x)
        cy = torch.tensor(y)
        out = torch.where(ct, cx, cy)
        return [out]

    op_tester.run(init_builder, reference, 'infer')


def test_where_5(op_tester):
    x = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 10])
    x = np.float32(x)
    y = np.float32(y)
    condition = np.array([False])

    def init_builder(builder):
        i1 = builder.addInputTensor(condition)
        i2 = builder.addInputTensor(x)
        i3 = builder.addInputTensor(y)
        o = builder.aiOnnx.where([i1, i2, i3])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        ct = torch.tensor(condition)
        cx = torch.tensor(x)
        cy = torch.tensor(y)
        out = torch.where(ct, cx, cy)
        return [out]

    op_tester.run(init_builder, reference, 'infer')


def test_where_grad0(op_tester):
    condition = np.array([[True, False], [True, True]], dtype=np.bool)
    x = np.array([[1, 2], [3, 4]], dtype=np.float32)
    y = np.array([[9, 8], [7, 6]], dtype=np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(condition)
        i2 = builder.addInputTensor(x)
        i3 = builder.addInputTensor(y)
        o = builder.aiOnnx.where([i1, i2, i3])
        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + i2,
            popart.reservedGradientPrefix() + i3,
            popart.reservedGradientPrefix() + o,
        ]

    def reference(ref_data):
        tc = torch.tensor(condition)
        tx = torch.tensor(x, requires_grad=True)
        ty = torch.tensor(y, requires_grad=True)
        out = torch.where(tc, tx, ty)
        d__o = ref_data.getOutputTensorGrad(0)
        out.backward(torch.tensor(d__o))
        return [out, tx.grad, ty.grad, None]

    op_tester.run(init_builder, reference, 'train')


def test_where_grad1(op_tester):
    condition = np.array([[True, False], [True, True]], dtype=np.bool)
    x = np.array([[1, 2]], dtype=np.float32)
    y = np.array([[9, 8], [7, 6]], dtype=np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(condition)
        i2 = builder.addInputTensor(x)
        i3 = builder.addInputTensor(y)
        o = builder.aiOnnx.where([i1, i2, i3])
        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + i2,
            popart.reservedGradientPrefix() + i3,
            popart.reservedGradientPrefix() + o,
        ]

    def reference(ref_data):
        tc = torch.tensor(condition)
        tx = torch.tensor(x, requires_grad=True)
        ty = torch.tensor(y, requires_grad=True)
        out = torch.where(tc, tx, ty)
        d__o = ref_data.getOutputTensorGrad(0)
        out.backward(torch.tensor(d__o))
        return [out, tx.grad, ty.grad, None]

    op_tester.run(init_builder, reference, 'train')


def test_where_grad2(op_tester):
    x = np.arange(9, dtype=np.float32)
    y = 10 * x
    condition = x < 5

    def init_builder(builder):
        i1 = builder.addInputTensor(condition)
        i2 = builder.addInputTensor(x)
        i3 = builder.addInputTensor(y)
        o = builder.aiOnnx.where([i1, i2, i3])
        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + i2,
            popart.reservedGradientPrefix() + i3,
            popart.reservedGradientPrefix() + o,
        ]

    def reference(ref_data):
        tc = torch.tensor(condition)
        tx = torch.tensor(x, requires_grad=True)
        ty = torch.tensor(y, requires_grad=True)
        out = torch.where(tc, tx, ty)
        d__o = ref_data.getOutputTensorGrad(0)
        out.backward(torch.tensor(d__o))
        return [out, tx.grad, ty.grad, None]

    op_tester.run(init_builder, reference, 'train')


def test_where_grad3(op_tester):
    x, y = np.ogrid[:3, :4]
    x = np.float32(x)
    y = np.float32(y)
    y = 10 + y
    condition = x < y

    def init_builder(builder):
        i1 = builder.addInputTensor(condition)
        i2 = builder.addInputTensor(x)
        i3 = builder.addInputTensor(y)
        o = builder.aiOnnx.where([i1, i2, i3])
        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + i2,
            popart.reservedGradientPrefix() + i3,
            popart.reservedGradientPrefix() + o,
        ]

    def reference(ref_data):
        tc = torch.tensor(condition)
        tx = torch.tensor(x, requires_grad=True)
        ty = torch.tensor(y, requires_grad=True)
        out = torch.where(tc, tx, ty)
        d__o = ref_data.getOutputTensorGrad(0)
        out.backward(torch.tensor(d__o))
        return [out, tx.grad, ty.grad, None]

    op_tester.run(init_builder, reference, 'train')


def test_where_grad4(op_tester):
    x = np.array([[0, 1, 2], [0, 2, 4], [0, 3, 6]])
    y = np.array([-1])
    x = np.float32(x)
    y = np.float32(y)
    condition = x < 4

    def init_builder(builder):
        i1 = builder.addInputTensor(condition)
        i2 = builder.addInputTensor(x)
        i3 = builder.addInputTensor(y)
        o = builder.aiOnnx.where([i1, i2, i3])
        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + i2,
            popart.reservedGradientPrefix() + i3,
            popart.reservedGradientPrefix() + o,
        ]

    def reference(ref_data):
        tc = torch.tensor(condition)
        tx = torch.tensor(x, requires_grad=True)
        ty = torch.tensor(y, requires_grad=True)
        out = torch.where(tc, tx, ty)
        d__o = ref_data.getOutputTensorGrad(0)
        out.backward(torch.tensor(d__o))
        return [out, tx.grad, ty.grad, None]

    op_tester.run(init_builder, reference, 'train')


def test_where_grad5(op_tester):
    x = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 10])
    x = np.float32(x)
    y = np.float32(y)
    condition = np.array([False])

    def init_builder(builder):
        i1 = builder.addInputTensor(condition)
        i2 = builder.addInputTensor(x)
        i3 = builder.addInputTensor(y)
        o = builder.aiOnnx.where([i1, i2, i3])
        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + i2,
            popart.reservedGradientPrefix() + i3,
            popart.reservedGradientPrefix() + o,
        ]

    def reference(ref_data):
        tc = torch.tensor(condition)
        tx = torch.tensor(x, requires_grad=True)
        ty = torch.tensor(y, requires_grad=True)
        out = torch.where(tc, tx, ty)
        d__o = ref_data.getOutputTensorGrad(0)
        out.backward(torch.tensor(d__o))
        return [out, tx.grad, ty.grad, None]

    op_tester.run(init_builder, reference, 'train')
