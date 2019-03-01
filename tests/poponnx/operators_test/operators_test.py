import numpy as np
import poponnx
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
    ops_public = poponnx.getSupportedOperations(False)
    assert (len(ops_public) > 0)

    ops_all = poponnx.getSupportedOperations(True)
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
        return [o, 'd__' + i1, 'd__' + i2, 'd__' + o]

    def reference(ref_data):
        t1 = torch.tensor(d1, requires_grad=True)
        t2 = torch.tensor(d2, requires_grad=True)
        out = t1 + t2
        d__o = ref_data.getOutputTensorGrad(0)
        out.backward(torch.tensor(d__o))
        return [out, t1.grad, t2.grad, None]

    op_tester.passes = ['PreUniRepl']
    op_tester.run(init_builder, reference, 'train')


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

    op_tester.passes = ["PreUniRepl"]
    op_tester.run(init_builder, reference)


def test_div(tmpdir):
    # create test data
    d1 = np.random.rand(4).astype(np.float32)
    d2 = np.random.rand(4).astype(np.float32)

    # create graph
    test = tu.BasicSession(tmpdir)
    i1 = test.add_input_tensor(d1)
    i2 = test.add_input_tensor(d2)
    o = test.builder.aiOnnx.div([i1, i2])
    test.builder.addOutputTensor(o)

    test.passes.extend(["PreUniRepl"])

    # run the poponnx session
    anchors = test.run(o, [o], 'infer')

    # create and run numpy reference
    def numpy_reference(i1, i2):
        outputs = {}
        outputs[o] = i1 / i2
        return outputs

    reference_results = numpy_reference(d1, d2)

    # compare results
    for key in [o]:
        print('Checking anchor %s ...' % (key, ))
        assert np.array_equal(anchors[key], reference_results[key])


def test_div_grad(tmpdir):
    # create test data
    d1 = np.random.rand(4, 1, 4).astype(np.float32)
    d2 = np.random.rand(3, 1).astype(np.float32)

    # create graph
    test = tu.BasicSession(tmpdir)
    i1 = test.add_input_tensor(d1)
    i2 = test.add_input_tensor(d2)
    o = test.builder.aiOnnx.div([i1, i2])
    test.builder.addOutputTensor(o)

    test.passes.extend(["PreUniRepl", "DivArg0GradOp", "DivArg1GradOp"])

    # run the poponnx session
    anchors = test.run(o, [o, 'd__' + o, 'd__' + i1, 'd__' + i2], 'train')

    # create and run torch reference
    def torch_reference(d__o):
        t1 = torch.tensor(d1, requires_grad=True)
        t2 = torch.tensor(d2, requires_grad=True)
        out = t1 / t2
        out.backward(torch.tensor(d__o))

        outputs = {}
        outputs[o] = out.data.numpy()
        outputs['d__' + i1] = t1.grad.data.numpy()
        outputs['d__' + i2] = t2.grad.data.numpy()
        return outputs

    reference_results = torch_reference(anchors['d__' + o])

    # compare results
    for key in [o, 'd__' + i1, 'd__' + i2]:
        print('Checking anchor %s ...' % (key, ))
        assert np.allclose(anchors[key], reference_results[key])


def test_reciprocal_grad(op_tester):
    # create test data
    d1 = np.random.rand(4).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.reciprocal([i1])
        builder.addOutputTensor(o)
        return [o, 'd__' + i1, 'd__' + o]

    def reference(ref_data):
        a = torch.tensor(d1, requires_grad=True)
        b = 1 / a
        d__o = ref_data.getOutputTensorGrad(0)
        b.backward(torch.tensor(d__o))
        return [b, a.grad, None]

    op_tester.passes = ['PreUniRepl', 'ReciprocalGradOp']
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

    op_tester.passes = ['PreUniRepl']
    op_tester.run(init_builder, reference, 'infer')


def test_sqrt_grad(op_tester):
    # create test data
    d1 = np.random.rand(4).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.sqrt([i1])
        builder.addOutputTensor(o)
        return [o, 'd__' + i1, 'd__' + o]

    def reference(ref_data):
        a = torch.tensor(d1, requires_grad=True)
        out = torch.sqrt(a)
        d__o = ref_data.getOutputTensorGrad(0)
        out.backward(torch.tensor(d__o))
        return [out, a.grad, None]

    op_tester.passes = ['PreUniRepl', 'SqrtGradOp']
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
        return [o, 'd__' + i1, 'd__' + o]

    def reference(ref_data):
        a = torch.tensor(d1, requires_grad=True)
        b = torch.exp(a)
        d__o = ref_data.getOutputTensorGrad(0)
        b.backward(torch.tensor(d__o))
        return [b, a.grad, None]

    op_tester.passes = ['PreUniRepl', 'ExpGradOp']
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
        return [o, 'd__' + i1, 'd__' + o]

    def reference(ref_data):
        a = torch.tensor(d1, requires_grad=True)
        b = torch.sigmoid(a)
        d__o = ref_data.getOutputTensorGrad(0)
        b.backward(torch.tensor(d__o))
        return [b, a.grad, None]

    op_tester.passes = ['PreUniRepl']
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
        return [o, 'd__' + i1, 'd__' + o]

    def reference(ref_data):
        a = torch.tensor(d1, requires_grad=True)
        o = a.permute(1, 3, 0, 4, 2)

        d__o = ref_data.getOutputTensorGrad(0)
        o.backward(torch.tensor(d__o))

        return [o, a.grad, None]

    op_tester.passes = ['PreUniRepl']
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
        return [o, 'd__' + i1, 'd__' + o]

    def reference(ref_data):
        a = torch.tensor(d1, requires_grad=True)
        b = torch.log(a)
        d__o = ref_data.getOutputTensorGrad(0)
        b.backward(torch.tensor(d__o))
        return [b, a.grad, None]

    op_tester.passes = ['PreUniRepl', 'LogGradOp']
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

    op_tester.passes = ['LogSoftmaxOp', 'LogGradOp']
    op_tester.run(init_builder, reference, 'infer')


def test_logsoftmax_grad(op_tester):
    # create test data
    d1 = np.random.rand(1, 4).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.logsoftmax([i1])
        builder.addOutputTensor(o)
        return [o, 'd__' + i1, 'd__' + o]

    def reference(ref_data):
        a = torch.tensor(d1, requires_grad=True)
        lsm = torch.nn.LogSoftmax(dim=1)
        b = lsm(a)
        d__o = ref_data.getOutputTensorGrad(0)
        b.backward(torch.tensor(d__o))
        return [b, a.grad, None]

    op_tester.passes = ['PreUniRepl', 'LogSoftmaxOp', 'LogGradOp']
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

    op_tester.run(init_builder, reference, 'infer')


def test_unsqueeze_grad(op_tester):
    d1 = np.random.rand(3, 4, 5).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.unsqueeze([i1], axes=[0, 4])
        builder.addOutputTensor(o)
        return [o, 'd__' + i1, 'd__' + o]

    def reference(ref_data):
        a = torch.tensor(d1, requires_grad=True)
        o = torch.unsqueeze(a, 0)
        o = torch.unsqueeze(o, 4)
        d__o = ref_data.getOutputTensorGrad(0)
        o.backward(torch.tensor(d__o))
        return [o, a.grad, None]

    op_tester.passes = ['PreUniRepl']
    op_tester.run(init_builder, reference, 'train')


def test_slice(op_tester):
    d1 = np.array([[1., 2., 3., 4.], [5., 6., 7., 8.]]).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.slice([i1], axes=[0, 1], starts=[1, 0], ends=[2, 3])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        o = d1[1:2, 0:3]

        return [o]

    op_tester.run(init_builder, reference, 'infer')


def test_slice_default_axes(op_tester):
    d1 = np.array([[1., 2., 3., 4.], [5., 6., 7., 8.]]).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.slice([i1], axes=[], starts=[1, 0], ends=[2, 3])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        o = d1[1:2, 0:3]

        return [o]

    op_tester.run(init_builder, reference, 'infer')


def test_slice_neg(op_tester):
    d1 = np.array([1., 2., 3., 4., 5., 6., 7., 8.]).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.slice([i1], axes=[0], starts=[-5], ends=[-3])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        o = d1[-5:-3]

        return [o]

    op_tester.run(init_builder, reference, 'infer')


def test_slice_grad(op_tester):
    d1 = np.array([[1., 2., 3., 4.], [5., 6., 7., 8.]]).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.slice([i1], axes=[0, 1], starts=[1, 0], ends=[2, 3])
        builder.addOutputTensor(o)
        return [o, 'd__' + i1, 'd__' + o]

    def reference(ref_data):
        a = torch.tensor(d1, requires_grad=True)
        o = a[1:2, 0:3]

        d__o = ref_data.getOutputTensorGrad(0)

        o.backward(torch.tensor(d__o))

        return [o, a.grad, None]

    op_tester.passes = ['PreUniRepl']
    op_tester.run(init_builder, reference, 'train')


def test_pad(op_tester):
    data = np.array([[[1., 2.], [3., 4.]]]).astype(np.float32)
    _test_pad(
        op_tester,
        data,
        lower_padding=(2, 1, 1),
        upper_padding=(1, 0, 2),
        mode='constant')


def test_pad_with_value(op_tester):
    data = np.array([[[1., 2.], [3., 4.]]]).astype(np.float32)
    _test_pad(
        op_tester,
        data,
        lower_padding=(2, 1, 1),
        upper_padding=(1, 0, 2),
        mode='constant',
        pad_value=0.3)


def test_pad_type_edge(op_tester):
    data = np.array([[[1., 2.], [3., 4.]]]).astype(np.float32)
    _test_pad(
        op_tester,
        data,
        lower_padding=(2, 1, 1),
        upper_padding=(1, 0, 2),
        mode='edge')


def test_pad_type_reflect(op_tester):
    data = np.array([[1., 2., 3.], [4., 5., 6.]]).astype(np.float32)
    _test_pad(
        op_tester,
        data,
        lower_padding=(1, 0),
        upper_padding=(0, 2),
        mode='reflect')


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

    op_tester.passes = ['PreUniRepl']
    op_tester.run(init_builder, reference, 'infer')


def test_pad_grad(op_tester):
    d1 = np.array([[1., 2., 3., 4.], [5., 6., 7., 8.]]).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.pad([i1], [0, 2, 1, 1], "constant", 0)
        builder.addOutputTensor(o)
        return [o, 'd__' + i1, 'd__' + o]

    def reference(ref_data):

        a = torch.tensor(d1, requires_grad=True)
        o = F.pad(input=a, pad=(2, 1, 0, 1), mode='constant', value=0)

        d__o = ref_data.getOutputTensorGrad(0)

        o.backward(torch.tensor(d__o))

        return [o, a.grad, d__o]

    op_tester.passes = ['PreUniRepl']
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
        return [o, 'd__' + i1, 'd__' + i3]

    def reference(ref_data):
        return [output, data, np.sign(updates) * 0.1]

    op_tester.passes = ['PreUniRepl']
    op_tester.run(init_builder, reference, 'train')


def test_scatter_1(op_tester):
    data = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]]).astype(np.float32)
    indices = np.array([[1, 3]]).astype(np.int32)
    updates = np.array([[-1.1, 2.1]]).astype(np.float32)
    output = np.array([[1.0, -1.1, 3.0, 2.1, 5.0]]).astype(np.float32)
    d_data = np.array([[0.1, 0, 0.1, 0, 0.1]]).astype(np.float32)
    axis = 1

    def init_builder(builder):
        i1 = builder.addInputTensor(data)
        i2 = builder.addInputTensor(indices)
        i3 = builder.addInputTensor(updates)
        o = builder.aiOnnx.scatter([i1, i2, i3], axis)
        builder.addOutputTensor(o)
        return [o, 'd__' + i1, 'd__' + i3]

    def reference(ref_data):
        return [output, d_data, np.sign(updates) * 0.1]

    op_tester.passes = ['PreUniRepl']
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
        return [o, 'd__' + i1, 'd__' + i3]

    def reference(ref_data):
        return [output, data, np.sign(updates) * 0.1]

    op_tester.passes = ['PreUniRepl']
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


def test_dropout_testing(op_tester):
    d1 = np.random.rand(2).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        (o, ) = builder.aiOnnx.dropout([i1], 1, 0.5, "test_dropout")
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        t1 = torch.tensor(d1)
        dropout = torch.nn.Dropout()
        dropout.eval()
        out = dropout(t1)

        return [out]

    op_tester.passes = ['OpToIdentity']
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

    op_tester.run(init_builder, reference, 'infer')


# Verify when in training mode that the exception is thrown
def test_dropout_training(op_tester):
    d1 = np.random.rand(2).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        (o, ) = builder.aiOnnx.dropout([i1], 1, 0.5, "test_dropout")
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        t1 = torch.tensor(d1)
        dropout = torch.nn.Dropout()
        out = dropout(t1)

        return [out]

    op_tester.passes = ['OpToIdentity']

    # Case 2 b does not have the size as x.dim(1)
    with pytest.raises(poponnx.poponnx_exception) as e_info:
        op_tester.run(init_builder, reference, 'train')

    assert (e_info.value.args[0] == "Dropout does not support training")


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
