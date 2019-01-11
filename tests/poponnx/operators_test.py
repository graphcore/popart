import numpy as np
import pytest
import poponnx
import torch
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
        o = builder.add([i1, i2], "test_add")
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
        o = builder.convolution([d, f], [1, 1], [1, 1, 1, 1], [1, 1], 1)
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


def test_matmul(op_tester):
    d1 = np.random.rand(2, 3).astype(np.float32)
    d2 = np.random.rand(3, 4).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        o = builder.matmul([i1, i2])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        out = np.matmul(d1, d2)
        return [out]

    op_tester.run(init_builder, reference)


def test_average_pool_1(op_tester):
    d1 = np.random.rand(1, 1, 6, 6).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.averagepool([i1], [2, 2], [2, 2], [0, 0, 0, 0])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        avgpool = torch.nn.AvgPool2d(
            kernel_size=2, stride=2, padding=0, count_include_pad=False)
        out = avgpool(torch.from_numpy(d1))
        return [out]

    op_tester.run(init_builder, reference)


def test_average_pool_2(op_tester):
    d1 = np.random.rand(1, 1, 14, 14).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.averagepool([i1], [3, 3], [1, 1], [1, 1, 1, 1])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        avgpool = torch.nn.AvgPool2d(
            kernel_size=3, stride=1, padding=1, count_include_pad=False)
        out = avgpool(torch.from_numpy(d1))
        return [out]

    op_tester.run(init_builder, reference)


def test_average_pool_3(op_tester):
    d1 = np.random.rand(1, 1, 14, 14).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.averagepool([i1], [3, 3], [2, 2], [0, 0, 0, 0])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        avgpool = torch.nn.AvgPool2d(
            kernel_size=3, stride=2, padding=0, count_include_pad=False)
        out = avgpool(torch.from_numpy(d1))
        return [out]

    op_tester.run(init_builder, reference)


def test_average_pool_with_count_include_pad(op_tester):

    builder = poponnx.Builder()

    i1 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1, 1, 14, 14]))
    o = builder.averagepool([i1], [3, 3], [2, 2], [0, 0, 0, 0])
    builder.addOutputTensor(o)

    builder.removeNodeAttribute("count_include_pad", set(o))
    builder.addNodeAttribute("count_include_pad", 1, set(o))

    dataFlow = poponnx.DataFlow(1, {o: poponnx.AnchorReturnType("ALL")})
    optimizer = poponnx.ConstSGD(0.01)
    losses = [poponnx.L1Loss(o, "l1LossVal", 0.1)]
    proto = builder.getModelProto()

    opts = poponnx.SessionOptionsCore()
    opts.logging = {'all': 'TRACE'}

    with pytest.raises(poponnx.poponnx_exception) as e_info:
        session = poponnx.Session(
            fnModel=proto,
            dataFeed=dataFlow,
            losses=losses,
            optimizer=optimizer,
            userOptions=opts)

    assert (e_info.value.args[0].startswith(
        "`count_include_pad` is not supported"))


def test_maxpool_1(op_tester):
    d1 = np.random.rand(1, 1, 16, 16).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.maxpool([i1], [2, 2], [2, 2], [0, 0, 0, 0])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        t1 = torch.tensor(d1, requires_grad=True)
        avgpool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        out = avgpool(t1)
        return [out]

    op_tester.run(init_builder, reference, step_type='infer')


def test_maxpool_2(op_tester):
    d1 = np.random.rand(1, 1, 16, 16).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.maxpool([i1], [3, 3], [1, 1], [1, 1, 1, 1])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        t1 = torch.tensor(d1, requires_grad=True)
        avgpool = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        out = avgpool(t1)
        return [out]

    op_tester.run(init_builder, reference, step_type='infer')


def test_maxpool_3(op_tester):
    d1 = np.random.rand(1, 1, 16, 16).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.maxpool([i1], [5, 5], [2, 2], [2, 2, 2, 2])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        t1 = torch.tensor(d1, requires_grad=True)
        avgpool = torch.nn.MaxPool2d(kernel_size=5, stride=2, padding=2)
        out = avgpool(t1)
        return [out]

    op_tester.run(init_builder, reference, step_type='infer')


def test_maxpool_4(op_tester):
    d1 = np.random.rand(1, 1, 16, 16).astype(np.float16)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.maxpool([i1], [5, 5], [2, 2], [2, 2, 2, 2])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        t1 = torch.tensor(d1.astype(np.float32), requires_grad=True)
        avgpool = torch.nn.MaxPool2d(kernel_size=5, stride=2, padding=2)
        out = avgpool(t1).data.numpy().astype(np.float16)
        return [out]

    op_tester.run(init_builder, reference, step_type='infer')


def test_maxpool_grad(op_tester):
    d1 = np.random.rand(1, 1, 6, 6).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.maxpool([i1], [2, 2], [2, 2], [0, 0, 0, 0])
        builder.addOutputTensor(o)
        return [o, 'd__' + i1, 'd__' + o]

    def reference(ref_data):
        t1 = torch.tensor(d1, requires_grad=True)
        avgpool = torch.nn.MaxPool2d(2, 2)
        out = avgpool(t1)
        d__o = ref_data.getOutputTensorGrad(0)
        out.backward(torch.tensor(d__o))
        return [out, t1.grad, None]

    op_tester.passes = ['PreUniRepl']
    op_tester.run(init_builder, reference, step_type='train')


def test_mul(op_tester):
    d1 = np.random.rand(2).astype(np.float32)
    d2 = np.random.rand(2).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        o = builder.mul([i1, i2])
        builder.addOutputTensor(o)
        return [o, 'd__' + i1, 'd__' + i2, 'd__' + o]

    def reference(ref_data):
        t1 = torch.tensor(d1, requires_grad=True)
        t2 = torch.tensor(d2, requires_grad=True)
        out = t1 * t2
        d__o = ref_data.getOutputTensorGrad(0)
        out.backward(torch.tensor(d__o))
        return [out, t1.grad, t2.grad, None]

    op_tester.passes = ['PreUniRepl', 'MulArgGradOp']
    op_tester.run(init_builder, reference, step_type='train')


def test_broadcast_mul(op_tester):
    d1 = np.random.rand(2, 2).astype(np.float32)
    d2 = np.random.rand(2).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        o = builder.mul([i1, i2])
        builder.addOutputTensor(o)
        return [o, 'd__' + i1, 'd__' + i2, 'd__' + o]

    def reference(ref_data):
        t1 = torch.tensor(d1, requires_grad=True)
        t2 = torch.tensor(d2, requires_grad=True)
        out = t1 * t2
        d__o = ref_data.getOutputTensorGrad(0)
        out.backward(torch.tensor(d__o))
        return [out, t1.grad, t2.grad, None]

    op_tester.passes = ['PreUniRepl', 'MulArgGradOp']
    op_tester.run(init_builder, reference, step_type='train')


def test_reciprocal(op_tester):
    # create test data
    d1 = np.random.rand(4).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.reciprocal([i1])
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
    o = test.builder.div([i1, i2])
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
    o = test.builder.div([i1, i2])
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
        o = builder.reciprocal([i1])
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


def test_sin(op_tester):
    # create test data
    d1 = np.random.rand(4).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.sin([i1])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        a = torch.tensor(d1, requires_grad=True)
        b = torch.sin(a)
        return [b]

    op_tester.run(init_builder, reference, 'infer')


def test_sin_grad(op_tester):
    # create test data
    d1 = np.random.rand(4).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.sin([i1])
        builder.addOutputTensor(o)
        return [o, 'd__' + i1, 'd__' + o]

    def reference(ref_data):
        a = torch.tensor(d1, requires_grad=True)
        b = torch.sin(a)
        d__o = ref_data.getOutputTensorGrad(0)
        b.backward(torch.tensor(d__o))
        return [b, a.grad, None]

    op_tester.passes = ['PreUniRepl', 'SinGradOp']
    op_tester.run(init_builder, reference, 'train')


def test_cos(op_tester):
    # create test data
    d1 = np.random.rand(4).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.cos([i1])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        a = torch.tensor(d1, requires_grad=True)
        b = torch.cos(a)
        return [b]

    op_tester.run(init_builder, reference, 'infer')


def test_cos_grad(op_tester):
    # create test data
    d1 = np.random.rand(4).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.cos([i1])
        builder.addOutputTensor(o)
        return [o, 'd__' + i1, 'd__' + o]

    def reference(ref_data):
        a = torch.tensor(d1, requires_grad=True)
        b = torch.cos(a)
        d__o = ref_data.getOutputTensorGrad(0)
        b.backward(torch.tensor(d__o))
        return [b, a.grad, None]

    op_tester.passes = ['PreUniRepl', 'CosGradOp']
    op_tester.run(init_builder, reference, 'train')


def test_tan(op_tester):
    # create test data
    d1 = np.random.rand(4).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.tan([i1])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        a = torch.tensor(d1, requires_grad=True)
        b = torch.tan(a)
        return [b]

    op_tester.passes = ['TanToSinOverCos']
    op_tester.run(init_builder, reference, 'infer')


def test_tan_grad(op_tester):
    # create test data
    d1 = np.random.rand(4).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.tan([i1])
        builder.addOutputTensor(o)
        return [o, 'd__' + i1, 'd__' + o]

    def reference(ref_data):
        a = torch.tensor(d1, requires_grad=True)
        b = torch.tan(a)
        d__o = ref_data.getOutputTensorGrad(0)
        b.backward(torch.tensor(d__o))
        return [b, a.grad, None]

    op_tester.passes = [
        'PreUniRepl', 'TanToSinOverCos', 'DivArg0GradOp', 'DivArg1GradOp',
        'SinGradOp', 'CosGradOp'
    ]
    op_tester.run(init_builder, reference, 'train')


def test_sqrt(op_tester):
    # create test data
    d1 = np.random.rand(4).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.sqrt([i1])
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
        o = builder.sqrt([i1])
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


def test_tanh(op_tester):
    # create test data
    d1 = np.random.rand(4).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.tanh([i1])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        a = torch.tensor(d1, requires_grad=True)
        b = torch.tanh(a)
        return [b]

    op_tester.run(init_builder, reference, 'infer')


def test_tanh_grad(op_tester):
    # create test data
    d1 = np.random.rand(4).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.tanh([i1])
        builder.addOutputTensor(o)
        return [o, 'd__' + i1, 'd__' + o]

    def reference(ref_data):
        a = torch.tensor(d1, requires_grad=True)
        b = torch.tanh(a)
        d__o = ref_data.getOutputTensorGrad(0)
        b.backward(torch.tensor(d__o))
        return [b, a.grad, None]

    op_tester.passes = ['PreUniRepl']
    op_tester.run(init_builder, reference, 'train')


def test_exp(op_tester):
    # create test data
    d1 = np.random.rand(4).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.exp([i1])
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
        o = builder.exp([i1])
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


def test_cosh(op_tester):
    # create test data
    d1 = np.random.rand(4).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.cosh([i1])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        a = torch.tensor(d1, requires_grad=True)
        b = torch.cosh(a)
        return [b]

    op_tester.passes = ['PreUniRepl', 'CoshOp']
    op_tester.run(init_builder, reference, 'infer')


def test_cosh_grad(op_tester):
    # create test data
    d1 = np.random.rand(4).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.cosh([i1])
        builder.addOutputTensor(o)
        return [o, 'd__' + i1, 'd__' + o]

    def reference(ref_data):
        a = torch.tensor(d1, requires_grad=True)
        b = torch.cosh(a)
        d__o = ref_data.getOutputTensorGrad(0)
        b.backward(torch.tensor(d__o))
        return [b, a.grad, None]

    op_tester.passes = ['PreUniRepl', 'CoshOp', 'ExpGradOp']
    op_tester.run(init_builder, reference, 'train')


def test_sigmoid(op_tester):
    # create test data
    d1 = np.random.rand(4).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.sigmoid([i1])
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
        o = builder.sigmoid([i1])
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


def test_softmax(op_tester):
    # create test data
    # Note: poplar implementation of softmax
    # requires outer 'batch' dimension
    d1 = np.random.rand(1, 4).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.softmax([i1])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        a = torch.tensor(d1, requires_grad=True)
        # 'dim' corresponds to dim index over which
        # to perform softmax
        lsm = torch.nn.Softmax(dim=1)
        b = lsm(a)
        return [b]

    op_tester.run(init_builder, reference, 'infer')


def test_softmax_grad(op_tester):
    # create test data
    d1 = np.random.rand(1, 4).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.softmax([i1])
        builder.addOutputTensor(o)
        return [o, 'd__' + i1, 'd__' + o]

    def reference(ref_data):
        a = torch.tensor(d1, requires_grad=True)
        sm = torch.nn.Softmax(dim=1)
        b = sm(a)
        d__o = ref_data.getOutputTensorGrad(0)
        b.backward(torch.tensor(d__o))
        return [b, a.grad, None]

    op_tester.passes = ['PreUniRepl']
    op_tester.run(init_builder, reference, 'train')


def subsample_helper(op_tester, input, strides, output, grad_ouput):
    # create test data
    d1 = input

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.subsample([i1], strides)
        builder.addOutputTensor(o)
        return [o, 'd__' + i1]

    def reference(ref_data):
        return [output, grad_ouput]

    op_tester.passes = ['PreUniRepl', 'SqrtGradOp']
    op_tester.run(init_builder, reference, 'train')


def test_subample1(op_tester):
    subsample_helper(
        op_tester,
        np.array([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]],
                 dtype=np.float32), [2, 2],
        np.array([[1, 3], [3, 5]], dtype=np.float32),
        np.array(
            [[0.1, 0, 0.1, 0], [0, 0, 0, 0], [0.1, 0, 0.1, 0], [0, 0, 0, 0]],
            dtype=np.float32))


def test_subample2(op_tester):
    subsample_helper(
        op_tester,
        np.array([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]],
                 dtype=np.float32), [1, 1],
        np.array([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]],
                 dtype=np.float32),
        np.array([[0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1],
                  [0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]],
                 dtype=np.float32))


def test_subample3(op_tester):
    subsample_helper(
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


def test_subample4(op_tester):
    subsample_helper(
        op_tester,
        np.array([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]],
                 dtype=np.float32), [3, 3],
        np.array([[1, 4], [4, 7]], dtype=np.float32),
        np.array(
            [[0.1, 0, 0, 0.1], [0, 0, 0, 0], [0, 0, 0, 0], [0.1, 0, 0, 0.1]],
            dtype=np.float32))


def test_subample5(op_tester):
    subsample_helper(
        op_tester,
        np.array([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]],
                 dtype=np.float32), [4, 4], np.array([[1]], dtype=np.float32),
        np.array([[0.1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                 dtype=np.float32))


# Test the error case where there is a 0 stride
def test_subample6(op_tester):

    with pytest.raises(poponnx.poponnx_exception) as e_info:

        subsample_helper(
            op_tester,
            np.array([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]],
                     dtype=np.float32), [4, 0],
            np.array([[1]], dtype=np.float32),
            np.array(
                [[0.1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                dtype=np.float32))

    assert (e_info.value.args[0].startswith(
        "Strides invalid. 0 stride at index 1"))


# Test the error case where there is a 0 stride
def test_subample6(op_tester):

    with pytest.raises(poponnx.poponnx_exception) as e_info:

        subsample_helper(
            op_tester,
            np.array([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]],
                     dtype=np.float32), [4, 0],
            np.array([[1]], dtype=np.float32),
            np.array(
                [[0.1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                dtype=np.float32))

    assert (e_info.value.args[0].startswith(
        "Strides invalid. 0 stride at index 1"))


# The calculation for running_mean & running_variance is different for onnx and pytorch
# ONNX : running_mean = running_mean * momentum + mean * (1 - momentum)
# PyTorch : running_mean = mean * momentum + running_mean * (1 - momentum)
#
# https://pytorch.org/docs/stable/nn.html?highlight=batchnorm2d#torch.nn.BatchNorm2d
# https://github.com/onnx/onnx/blob/master/docs/Operators.md#BatchNormalization


def test_batchnorm_train_0_errorcases(op_tester):
    # create test data
    d1 = np.array([[[[1, 1], [1, 1]], [[1, 1], [1, 1]]],
                   [[[1, 0], [0, 1]], [[1, 0], [0, 1]]]],
                  dtype=np.float32)

    d2 = np.array([[1, 1], [1, 1]], dtype=np.float32)

    scale = np.ones(2).astype(np.float32)
    scale_2 = np.ones(1).astype(np.float32)
    b = np.zeros(2).astype(np.float32)
    b_2 = np.zeros(1).astype(np.float32)
    mean = np.zeros(2).astype(np.float32)
    mean_2 = np.zeros(1).astype(np.float32)
    var = np.ones(2).astype(np.float32)
    var_2 = np.ones(1).astype(np.float32)
    epsilon = 1e-05
    momentum = 0.1

    def init_builder_case0(builder):

        i1 = builder.addInputTensor(d2)
        iScale = builder.addInputTensor(scale)
        iB = builder.addInputTensor(b)
        iMean = builder.addInputTensor(mean)
        iVar = builder.addInputTensor(var)
        o = builder.batchnormalizationTraining(i1, iScale, iB, iMean, iVar,
                                               epsilon, momentum)

        return [o.y]

    def init_builder_case1(builder):

        i1 = builder.addInputTensor(d1)
        iScale = builder.addInputTensor(scale_2)
        iB = builder.addInputTensor(b)
        iMean = builder.addInputTensor(mean)
        iVar = builder.addInputTensor(var)
        o = builder.batchnormalizationTraining(i1, iScale, iB, iMean, iVar,
                                               epsilon, momentum)

        return [o.y]

    def init_builder_case2(builder):

        i1 = builder.addInputTensor(d1)
        iScale = builder.addInputTensor(scale)
        iB = builder.addInputTensor(b_2)
        iMean = builder.addInputTensor(mean)
        iVar = builder.addInputTensor(var)
        o = builder.batchnormalizationTraining(i1, iScale, iB, iMean, iVar,
                                               epsilon, momentum)

        return [o.y]

    def init_builder_case3(builder):

        i1 = builder.addInputTensor(d1)
        iScale = builder.addInputTensor(scale)
        iB = builder.addInputTensor(b)
        iMean = builder.addInputTensor(mean_2)
        iVar = builder.addInputTensor(var)
        o = builder.batchnormalizationTraining(i1, iScale, iB, iMean, iVar,
                                               epsilon, momentum)

        return [o.y]

    def init_builder_case4(builder):

        i1 = builder.addInputTensor(d1)
        iScale = builder.addInputTensor(scale)
        iB = builder.addInputTensor(b)
        iMean = builder.addInputTensor(mean)
        iVar = builder.addInputTensor(var_2)
        o = builder.batchnormalizationTraining(i1, iScale, iB, iMean, iVar,
                                               epsilon, momentum)

        return [o.y]

    op_tester.passes = ['PreUniRepl', 'ReciprocalGradOp']

    # Case 0 input tensor has less than 4 dimensions
    with pytest.raises(poponnx.poponnx_exception) as e_info:
        op_tester.run(init_builder_case0, None, 'train')

    assert (
        e_info.value.args[0] == "batch norm requires a rank > 4. x has rank 2")

    # Case 1 scale does not have the size as x.dim(1)
    with pytest.raises(poponnx.poponnx_exception) as e_info:
        op_tester.run(init_builder_case1, None, 'train')

    assert (
        e_info.value.args[0] ==
        "batch norm scale dimension 0 (1) does not equal x dimension 1 (2)")

    # Case 2 b does not have the size as x.dim(1)
    with pytest.raises(poponnx.poponnx_exception) as e_info:
        op_tester.run(init_builder_case2, None, 'train')

    assert (e_info.value.args[0] ==
            "batch norm b dimension 0 (1) does not equal x dimension 1 (2)")

    # Case 3 mean does not have the size as x.dim(1)
    with pytest.raises(poponnx.poponnx_exception) as e_info:
        op_tester.run(init_builder_case3, None, 'train')

    assert (e_info.value.args[0] ==
            "batch norm mean dimension 0 (1) does not equal x dimension 1 (2)")

    # Case 4 var does not have the size as x.dim(1)
    with pytest.raises(poponnx.poponnx_exception) as e_info:
        op_tester.run(init_builder_case4, None, 'train')

    assert (e_info.value.args[0] ==
            "batch norm var dimension 0 (1) does not equal x dimension 1 (2)")


def test_batchnorm_train_0(op_tester):
    # create test data
    d1 = np.array([[[[1, 1], [1, 1]], [[1, 1], [1, 1]]],
                   [[[1, 0], [0, 1]], [[1, 0], [0, 1]]]],
                  dtype=np.float32)

    scale = np.ones(2).astype(np.float32)
    b = np.zeros(2).astype(np.float32)
    mean = np.zeros(2).astype(np.float32)
    var = np.ones(2).astype(np.float32)
    epsilon = 1e-05
    momentum = 0.1

    def init_builder(builder):

        i1 = builder.addInputTensor(d1)
        iScale = builder.addInputTensor(scale)
        iB = builder.addInputTensor(b)
        iMean = builder.addInputTensor(mean)
        iVar = builder.addInputTensor(var)
        o = builder.batchnormalizationTraining(i1, iScale, iB, iMean, iVar,
                                               epsilon, momentum)

        #for x in range(5):
        #    o = builder.batchnormalizationTraining(o.y, iScale, iB, o.mean,
        #                                           o.var, epsilon, momentum)

        builder.addOutputTensor(o.y)
        builder.addOutputTensor(o.mean)
        builder.addOutputTensor(o.var)

        return [o.y, 'd__' + i1, 'd__' + o.y]

    def reference(ref_data):
        _input = torch.tensor(d1, requires_grad=True)
        _weight = torch.tensor(scale, requires_grad=True)
        _bias = torch.tensor(b, requires_grad=True)
        _mean = torch.tensor(mean, requires_grad=False)
        _var = torch.tensor(var, requires_grad=False)

        m = torch.nn.BatchNorm2d(
            2, eps=epsilon, momentum=momentum, track_running_stats=True)
        m.state_dict()['weight'].copy_(_weight)
        m.state_dict()['bias'].copy_(_bias)
        m.state_dict()['running_mean'].copy_(_mean)
        m.state_dict()['running_var'].copy_(_var)

        m.train()
        _y = m(_input)

        #for x in range(5):
        #    _y = m(_y)

        d__o = ref_data.getOutputTensorGrad(0)
        _y.backward(torch.tensor(d__o))

        return [_y, _input.grad, d__o]

    op_tester.passes = ['PreUniRepl', 'ReciprocalGradOp']
    op_tester.run(init_builder, reference, 'train')


def test_batchnorm_train_1(op_tester):
    # create test data
    d1 = np.random.rand(2, 2, 2, 2).astype(np.float32)
    scale = np.random.rand(2).astype(np.float32)
    b = np.random.rand(2).astype(np.float32)
    mean = np.zeros(2).astype(np.float32)
    var = np.ones(2).astype(np.float32)
    epsilon = 1e-05
    momentum = 0.1

    # Relax the relative tolerance as small numbers lose precison
    op_tester.rtol = 1e-04

    def init_builder(builder):

        i1 = builder.addInputTensor(d1)
        iScale = builder.addInputTensor(scale)
        iB = builder.addInputTensor(b)
        iMean = builder.addInputTensor(mean)
        iVar = builder.addInputTensor(var)
        o = builder.batchnormalizationTraining(i1, iScale, iB, iMean, iVar,
                                               epsilon, momentum)

        #for x in range(5):
        #    o = builder.batchnormalizationTraining(o.y, iScale, iB, o.mean,
        #                                           o.var, epsilon, momentum)

        builder.addOutputTensor(o.y)
        builder.addOutputTensor(o.mean)
        builder.addOutputTensor(o.var)
        return [o.y, 'd__' + i1, 'd__' + o.y]

    def reference(ref_data):
        _input = torch.tensor(d1, requires_grad=False)
        _weight = torch.tensor(scale, requires_grad=False)
        _bias = torch.tensor(b, requires_grad=False)
        _mean = torch.tensor(mean, requires_grad=False)
        _var = torch.tensor(var, requires_grad=False)

        m = torch.nn.BatchNorm2d(
            2, eps=epsilon, momentum=momentum, track_running_stats=True)
        m.state_dict()['weight'].copy_(_weight)
        m.state_dict()['bias'].copy_(_bias)
        m.state_dict()['running_mean'].copy_(_mean)
        m.state_dict()['running_var'].copy_(_var)

        m.train()
        _y = m(_input)

        #for x in range(5):
        #    _y = m(_y)

        _mean = m.state_dict()['running_mean']
        _var = m.state_dict()['running_var']

        d__o = ref_data.getOutputTensorGrad(0)
        _y.backward(torch.tensor(d__o))

        return [_y, _input.grad, d__o]

    op_tester.passes = ['PreUniRepl', 'ReciprocalGradOp']
    op_tester.run(init_builder, reference, 'train')


def test_batchnorm_train_2(op_tester):
    # create test data
    d1 = np.random.rand(2, 2, 2, 2, 2).astype(np.float32)

    scale = np.random.rand(2).astype(np.float32)
    b = np.random.rand(2).astype(np.float32)
    mean = np.zeros(2).astype(np.float32)
    var = np.ones(2).astype(np.float32)
    epsilon = 1e-05
    momentum = 0.1

    # Relax the relative tolerance as small numbers lose precison
    op_tester.rtol = 1e-04

    def init_builder(builder):

        i1 = builder.addInputTensor(d1)
        iScale = builder.addInputTensor(scale)
        iB = builder.addInputTensor(b)
        iMean = builder.addInputTensor(mean)
        iVar = builder.addInputTensor(var)
        o = builder.batchnormalizationTraining(i1, iScale, iB, iMean, iVar,
                                               epsilon, momentum)

        for x in range(15):
            o = builder.batchnormalizationTraining(o.y, iScale, iB, o.mean,
                                                   o.var, epsilon, momentum)

        builder.addOutputTensor(o.y)
        builder.addOutputTensor(o.mean)
        builder.addOutputTensor(o.var)
        return [o.y, o.mean, o.var]

    def reference(ref_data):
        _input = torch.tensor(d1, requires_grad=False)
        _weight = torch.tensor(scale, requires_grad=False)
        _bias = torch.tensor(b, requires_grad=False)
        _mean = torch.tensor(mean, requires_grad=False)
        _var = torch.tensor(var, requires_grad=False)

        m = torch.nn.BatchNorm3d(
            2, eps=epsilon, momentum=momentum, track_running_stats=True)
        m.state_dict()['weight'].copy_(_weight)
        m.state_dict()['bias'].copy_(_bias)
        m.state_dict()['running_mean'].copy_(_mean)
        m.state_dict()['running_var'].copy_(_var)

        m.train()
        _y = m(_input)

        for x in range(15):
            _y = m(_y)

        _mean = m.state_dict()['running_mean']
        _var = m.state_dict()['running_var']

        return [_y, None, None]

    op_tester.passes = ['PreUniRepl', 'ReciprocalGradOp']
    op_tester.run(init_builder, reference, 'train')


# This test does not work as the inputs are now rejects as the mean/var do not match
# input.{C}
# def test_batchnorm_train_3(op_tester):
#     # create test data
#     d1 = np.random.rand(0, 0, 0, 0).astype(np.float32)
#     scale = np.random.rand(0).astype(np.float32)
#     b = np.random.rand(0).astype(np.float32)
#     mean = np.zeros(1).astype(np.float32)
#     var = np.ones(1).astype(np.float32)
#     epsilon = 1e-05
#     momentum = 0.1

#     def init_builder(builder):

#         i1 = builder.addInputTensor(d1)
#         iScale = builder.addInputTensor(scale)
#         iB = builder.addInputTensor(b)
#         iMean = builder.addInputTensor(mean)
#         iVar = builder.addInputTensor(var)
#         o = builder.batchnormalizationTraining(i1, iScale, iB, iMean, iVar,
#                                                epsilon, momentum)
#         builder.addOutputTensor(o.y)
#         builder.addOutputTensor(o.mean)
#         builder.addOutputTensor(o.var)
#         return [o.y, o.mean, o.var]

#     def reference(ref_data):
#         _input = torch.tensor(d1, requires_grad=False)

#         return [_input, None, None]

#     op_tester.passes = ['PreUniRepl', 'ReciprocalGradOp']
#     op_tester.run(init_builder, reference, 'train')


def test_batchnorm_test_0(op_tester):
    # create test data
    d1 = np.array([[[[1, 1], [1, 1]], [[1, 1], [1, 1]]],
                   [[[1, 0], [0, 1]], [[1, 0], [0, 1]]]],
                  dtype=np.float32)

    scale = np.ones(2).astype(np.float32)
    b = np.zeros(2).astype(np.float32)
    mean = np.zeros(2).astype(np.float32)
    var = np.ones(2).astype(np.float32)
    epsilon = 1e-05
    momentum = 0.1

    def init_builder(builder):

        i1 = builder.addInputTensor(d1)
        iScale = builder.addInputTensor(scale)
        iB = builder.addInputTensor(b)
        iMean = builder.addInputTensor(mean)
        iVar = builder.addInputTensor(var)
        y = builder.batchnormalizationTesting(i1, iScale, iB, iMean, iVar,
                                              epsilon, momentum)
        builder.addOutputTensor(y)
        return [y]

    def reference(ref_data):
        _input = torch.tensor(d1, requires_grad=False)
        _weight = torch.tensor(scale, requires_grad=False)
        _bias = torch.tensor(b, requires_grad=False)
        _mean = torch.tensor(mean, requires_grad=False)
        _var = torch.tensor(var, requires_grad=False)

        m = torch.nn.BatchNorm2d(
            2, eps=epsilon, momentum=momentum, track_running_stats=True)
        m.state_dict()['weight'].copy_(_weight)
        m.state_dict()['bias'].copy_(_bias)
        m.state_dict()['running_mean'].copy_(_mean)
        m.state_dict()['running_var'].copy_(_var)

        m.eval()

        _y = m(_input)

        return [_y]

    op_tester.passes = ['PreUniRepl', 'ReciprocalGradOp']
    op_tester.run(init_builder, reference, 'infer')


def test_batchnorm_test_1(op_tester):
    # create test data
    d1 = np.random.rand(2, 2, 2, 2).astype(np.float32)
    scale = np.random.rand(2).astype(np.float32)
    b = np.random.rand(2).astype(np.float32)
    mean = np.zeros(2).astype(np.float32)
    var = np.ones(2).astype(np.float32)
    epsilon = 1e-05
    momentum = 0.1

    def init_builder(builder):

        i1 = builder.addInputTensor(d1)
        iScale = builder.addInputTensor(scale)
        iB = builder.addInputTensor(b)
        iMean = builder.addInputTensor(mean)
        iVar = builder.addInputTensor(var)
        y = builder.batchnormalizationTesting(i1, iScale, iB, iMean, iVar,
                                              epsilon, momentum)
        builder.addOutputTensor(y)
        return [y]

    def reference(ref_data):
        _input = torch.tensor(d1, requires_grad=False)
        _weight = torch.tensor(scale, requires_grad=False)
        _bias = torch.tensor(b, requires_grad=False)
        _mean = torch.tensor(mean, requires_grad=False)
        _var = torch.tensor(var, requires_grad=False)

        m = torch.nn.BatchNorm2d(
            2, eps=epsilon, momentum=momentum, track_running_stats=True)
        m.state_dict()['weight'].copy_(_weight)
        m.state_dict()['bias'].copy_(_bias)
        m.state_dict()['running_mean'].copy_(_mean)
        m.state_dict()['running_var'].copy_(_var)

        m.eval()

        _y = m(_input)

        return [_y]

    op_tester.passes = ['PreUniRepl', 'ReciprocalGradOp']
    op_tester.run(init_builder, reference, 'infer')


def test_batchnorm_test_2(op_tester):
    # create test data
    d1 = np.random.rand(2, 2, 2, 2, 2).astype(np.float32)
    scale = np.random.rand(2).astype(np.float32)
    b = np.random.rand(2).astype(np.float32)
    mean = np.zeros(2).astype(np.float32)
    var = np.ones(2).astype(np.float32)
    epsilon = 1e-05
    momentum = 0.1

    def init_builder(builder):

        i1 = builder.addInputTensor(d1)
        iScale = builder.addInputTensor(scale)
        iB = builder.addInputTensor(b)
        iMean = builder.addInputTensor(mean)
        iVar = builder.addInputTensor(var)
        y = builder.batchnormalizationTesting(i1, iScale, iB, iMean, iVar,
                                              epsilon, momentum)
        builder.addOutputTensor(y)
        return [y]  #o.mean, o.var]#, 'd__' + i1] # 'd__' + o.y]

    def reference(ref_data):
        _input = torch.tensor(d1, requires_grad=False)
        _weight = torch.tensor(scale, requires_grad=False)
        _bias = torch.tensor(b, requires_grad=False)
        _mean = torch.tensor(mean, requires_grad=False)
        _var = torch.tensor(var, requires_grad=False)

        m = torch.nn.BatchNorm3d(
            2, eps=epsilon, momentum=momentum, track_running_stats=True)
        m.state_dict()['weight'].copy_(_weight)
        m.state_dict()['bias'].copy_(_bias)
        m.state_dict()['running_mean'].copy_(_mean)
        m.state_dict()['running_var'].copy_(_var)

        m.eval()
        _y = m(_input)

        return [_y]

    op_tester.passes = ['PreUniRepl', 'ReciprocalGradOp']
    op_tester.run(init_builder, reference, 'infer')


def test_batchnorm_test_3(op_tester):
    # create test data
    d1 = np.random.rand(0, 0, 0, 0).astype(np.float32)
    scale = np.random.rand(0).astype(np.float32)
    b = np.random.rand(0).astype(np.float32)
    mean = np.zeros(0).astype(np.float32)
    var = np.ones(0).astype(np.float32)
    epsilon = 1e-05
    momentum = 0.1

    def init_builder(builder):

        i1 = builder.addInputTensor(d1)
        iScale = builder.addInputTensor(scale)
        iB = builder.addInputTensor(b)
        iMean = builder.addInputTensor(mean)
        iVar = builder.addInputTensor(var)
        y = builder.batchnormalizationTesting(i1, iScale, iB, iMean, iVar,
                                              epsilon, momentum)
        builder.addOutputTensor(y)
        return [y]

    def reference(ref_data):
        # In the case the output should match in the input,
        # torch does not like a all zero input
        _input = torch.tensor(d1, requires_grad=False)
        return [_input]

    op_tester.passes = ['PreUniRepl', 'ReciprocalGradOp']
    op_tester.check_shapes = False
    op_tester.run(init_builder, reference, 'infer')


def test_gemm_basic(op_tester):
    A = np.random.rand(2, 4).astype(np.float32)
    B = np.random.rand(4, 6).astype(np.float32)
    C = np.random.rand(2, 6).astype(np.float32)
    _test_gemm(op_tester, A, B, C, 1.0, 1.0, False, False)


def test_gemm_scale(op_tester):
    A = np.random.rand(2, 4).astype(np.float32)
    B = np.random.rand(4, 6).astype(np.float32)
    C = np.random.rand(2, 6).astype(np.float32)
    alpha = np.random.random(1).astype(np.float32)[0]
    beta = np.random.random(1).astype(np.float32)[0]
    _test_gemm(op_tester, A, B, C, alpha, beta, False, False)


def test_gemm_transpose_a(op_tester):
    A = np.random.rand(4, 2).astype(np.float32)
    B = np.random.rand(4, 6).astype(np.float32)
    C = np.random.rand(2, 6).astype(np.float32)
    _test_gemm(op_tester, A, B, C, 1.0, 1.0, True, False)


def test_gemm_transpose_b(op_tester):
    A = np.random.rand(2, 4).astype(np.float32)
    B = np.random.rand(6, 4).astype(np.float32)
    C = np.random.rand(2, 6).astype(np.float32)
    _test_gemm(op_tester, A, B, C, 1.0, 1.0, False, True)


def test_gemm_transpose_ab(op_tester):
    A = np.random.rand(4, 2).astype(np.float32)
    B = np.random.rand(6, 4).astype(np.float32)
    C = np.random.rand(2, 6).astype(np.float32)
    _test_gemm(op_tester, A, B, C, 1.0, 1.0, True, True)


def test_gemm_basic_grad(op_tester):
    A = np.random.rand(2, 4).astype(np.float32)
    B = np.random.rand(4, 6).astype(np.float32)
    C = np.random.rand(2, 6).astype(np.float32)
    _test_gemm_grad(op_tester, A, B, C, 1.0, 1.0, False, False)


def test_gemm_grad(op_tester):
    A = np.random.rand(4, 2).astype(np.float32)
    B = np.random.rand(6, 4).astype(np.float32)
    C = np.random.rand(2, 6).astype(np.float32)
    _test_gemm_grad(op_tester, A, B, C, 1.0, 1.0, True, True)


def test_gemm_grad_scale(op_tester):
    A = np.random.rand(2, 4).astype(np.float32)
    B = np.random.rand(4, 6).astype(np.float32)
    C = np.random.rand(2, 6).astype(np.float32)
    alpha = np.random.random(1).astype(np.float32)[0]
    beta = np.random.random(1).astype(np.float32)[0]
    _test_gemm_grad(op_tester, A, B, C, alpha, beta, False, False)


def _test_gemm(op_tester, A, B, C, alpha, beta, transA, transB):
    def init_builder(builder):
        i1 = builder.addInputTensor(A)
        i2 = builder.addInputTensor(B)
        i3 = builder.addInputTensor(C)
        o = builder.gemm([i1, i2, i3], alpha, beta, transA, transB)
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        a = A
        b = B
        c = C

        if transA:
            a = np.transpose(a)
        if transB:
            b = np.transpose(b)

        o = alpha * np.dot(a, b) + beta * c
        return [o]

    op_tester.passes = ['GemmDecomposition']
    op_tester.run(init_builder, reference, 'infer')


def _test_gemm_grad(op_tester, A, B, C, alpha, beta, transA, transB):
    alpha = float(alpha)
    beta = float(beta)

    def init_builder(builder):
        i1 = builder.addInputTensor(A)
        i2 = builder.addInputTensor(B)
        i3 = builder.addInputTensor(C)
        o = builder.gemm([i1, i2, i3], alpha, beta, transA, transB)
        builder.addOutputTensor(o)
        return [o, 'd__' + i1, 'd__' + i2, 'd__' + i3, 'd__' + o]

    def reference(ref_data):
        a = torch.tensor(A, requires_grad=True)
        b = torch.tensor(B, requires_grad=True)
        c = torch.tensor(C, requires_grad=True)

        if transA:
            a = a.permute(1, 0)
        if transB:
            b = b.permute(1, 0)

        o = alpha * torch.matmul(a, b) + beta * c
        d__o = ref_data.getOutputTensorGrad(0)
        o.backward(torch.tensor(d__o))
        return [o, a.grad, b.grad, c.grad, None]

    op_tester.passes = ['GemmDecomposition', 'PreUniRepl']
    op_tester.run(init_builder, reference, 'train')


def test_transpose(op_tester):
    d1 = np.random.rand(3, 5, 2, 7).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.transpose([i1], [2, 0, 3, 1])
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
        o = builder.transpose([i1], [1, 3, 0, 4, 2])
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
        o = builder.log([i1])
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
        o = builder.log([i1])
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
        o = builder.logsoftmax([i1])
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
        o = builder.logsoftmax([i1])
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


def test_squeeze(op_tester):
    d1 = np.random.rand(2, 1, 3, 4, 1).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.squeeze([i1], axes=[])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        o = np.squeeze(d1)
        return [o]

    op_tester.run(init_builder, reference, 'infer')


def test_squeeze_limited(op_tester):
    d1 = np.random.rand(2, 1, 3, 4, 1).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.squeeze([i1], axes=[1])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        o = np.squeeze(d1, axis=1)
        return [o]

    op_tester.run(init_builder, reference, 'infer')


def test_squeeze_unsorted_axes(op_tester):
    d1 = np.random.rand(2, 1, 3, 1, 4, 1).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.squeeze([i1], axes=[5, 1])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        o = np.squeeze(d1, axis=5)
        o = np.squeeze(o, axis=1)
        return [o]

    op_tester.run(init_builder, reference, 'infer')


def test_squeeze_grad(op_tester):
    d1 = np.random.rand(2, 1, 3, 4, 1).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.squeeze([i1], axes=[])
        builder.addOutputTensor(o)
        return [o, 'd__' + i1, 'd__' + o]

    def reference(ref_data):
        i1 = torch.tensor(d1, requires_grad=True)
        o = torch.squeeze(i1)
        d__o = ref_data.getOutputTensorGrad(0)
        o.backward(torch.tensor(d__o))
        return [o, i1.grad, None]

    op_tester.passes = ['PreUniRepl']
    op_tester.run(init_builder, reference, 'train')


def test_squeeze_limited_grad(op_tester):
    d1 = np.random.rand(2, 1, 3, 4, 1).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.squeeze([i1], axes=[1])
        builder.addOutputTensor(o)
        return [o, 'd__' + i1, 'd__' + o]

    def reference(ref_data):
        i1 = torch.tensor(d1, requires_grad=True)
        o = torch.squeeze(i1, dim=1)
        d__o = ref_data.getOutputTensorGrad(0)
        o.backward(torch.tensor(d__o))
        return [o, i1.grad, None]

    op_tester.passes = ['PreUniRepl']
    op_tester.run(init_builder, reference, 'train')


def test_unsqueeze(op_tester):
    d1 = np.random.rand(3, 4, 5).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.unsqueeze([i1], axes=[0, 4])
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
        o = builder.unsqueeze([i1], axes=[0, 4])
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


class LSTM_Helper():
    def __init__(self, **params):  # type: (*Any) -> None
        # LSTM Input Names
        X = str('X')
        W = str('W')
        R = str('R')
        B = str('B')
        H_0 = str('initial_h')
        C_0 = str('initial_c')
        P = str('P')
        number_of_gates = 4
        number_of_peepholes = 3

        required_inputs = [X, W, R]
        for i in required_inputs:
            assert i in params, "Missing Required Input: {0}".format(i)

        self.num_directions = params[W].shape[0]

        if self.num_directions == 1:
            for k in params.keys():
                if k != X:
                    params[k] = np.squeeze(params[k], axis=0)

            hidden_size = params[R].shape[-1]
            batch_size = params[X].shape[1]

            b = params[B] if B in params else np.zeros(
                2 * number_of_gates * hidden_size, dtype=np.float32)
            p = params[P] if P in params else np.zeros(
                number_of_peepholes * hidden_size, dtype=np.float32)
            h_0 = params[H_0] if H_0 in params else np.zeros(
                (batch_size, hidden_size), dtype=np.float32)
            c_0 = params[C_0] if C_0 in params else np.zeros(
                (batch_size, hidden_size), dtype=np.float32)

            self.X = params[X]
            self.W = params[W]
            self.R = params[R]
            self.B = b
            self.P = p
            self.H_0 = h_0
            self.C_0 = c_0
        else:
            raise NotImplementedError()

    def f(self, x):  # type: (np.ndarray) -> np.ndarray
        return 1 / (1 + np.exp(-x))

    def g(self, x):  # type: (np.ndarray) -> np.ndarray
        return np.tanh(x)

    def h(self, x):  # type: (np.ndarray) -> np.ndarray
        return np.tanh(x)

    def step(self):  # type: () -> Tuple[np.ndarray, np.ndarray]
        [p_i, p_o, p_f] = np.split(self.P, 3)
        h_list = []
        H_t = self.H_0
        C_t = self.C_0
        for x in np.split(self.X, self.X.shape[0], axis=0):
            gates = np.dot(x, np.transpose(self.W)) + np.dot(
                H_t, np.transpose(self.R)) + np.add(*np.split(self.B, 2))
            i, o, f, c = np.split(gates, 4, -1)
            i = self.f(i + p_i * C_t)
            f = self.f(f + p_f * C_t)
            c = self.g(c)
            C = f * C_t + i * c
            o = self.f(o + p_o * C)
            H = o * self.h(C)
            h_list.append(H)
            H_t = H
            C_t = C
        concatenated = np.concatenate(h_list)
        if self.num_directions == 1:
            output = np.expand_dims(concatenated, 1)
        return output, h_list[-1], C_t


def test_lstm(op_tester):
    d1 = np.array([[[1., 2., 3.], [4., 5., 6.]],
                   [[7., 8., 9.], [10., 11., 12.]]]).astype(np.float32)

    seq_length = d1.shape[0]
    batch_size = d1.shape[1]
    input_size = d1.shape[2]
    hidden_size = 7
    num_directions = 1

    d2 = np.random.rand(1, 4 * hidden_size, input_size).astype(np.float32)
    d3 = np.zeros((1, 4 * hidden_size, hidden_size)).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        i3 = builder.addInputTensor(d3)
        Y, Y_h, Y_c = builder.lstm([i1, i2, i3])
        builder.addOutputTensor(Y_h)
        return [Y, Y_h, Y_c]

    def reference(ref_data):
        lstm = LSTM_Helper(X=d1, W=d2, R=d3)
        Y, Y_h, Y_c = lstm.step()

        return [Y, Y_h, Y_c]

    op_tester.run(init_builder, reference, 'infer')


def test_lstm_biases(op_tester):
    d1 = np.array([[[1., 2., 3.], [4., 5., 6.]],
                   [[7., 8., 9.], [10., 11., 12.]]]).astype(np.float32)

    seq_length = d1.shape[0]
    batch_size = d1.shape[1]
    input_size = d1.shape[2]
    hidden_size = 5
    num_directions = 1

    d2 = np.random.rand(num_directions, 4 * hidden_size,
                        input_size).astype(np.float32)
    d3 = np.random.rand(num_directions, 4 * hidden_size,
                        hidden_size).astype(np.float32)
    d4 = np.random.rand(num_directions, 8 * hidden_size).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        i3 = builder.addInputTensor(d3)
        i4 = builder.addInputTensor(d4)
        Y, Y_h, Y_c = builder.lstm([i1, i2, i3, i4])
        builder.addOutputTensor(Y_h)
        return [Y, Y_h, Y_c]

    def reference(ref_data):
        lstm = LSTM_Helper(X=d1, W=d2, R=d3, B=d4)
        Y, Y_h, Y_c = lstm.step()

        return [Y, Y_h, Y_c]

    op_tester.run(init_builder, reference, 'infer')


def test_lstm_initial_hc(op_tester):
    d1 = np.array([[[1., 2., 3.]]]).astype(np.float32)

    seq_length = d1.shape[0]
    batch_size = d1.shape[1]
    input_size = d1.shape[2]
    hidden_size = 2
    num_directions = 1

    d2 = np.random.rand(num_directions, 4 * hidden_size,
                        input_size).astype(np.float32)
    d3 = np.random.rand(num_directions, 4 * hidden_size,
                        hidden_size).astype(np.float32)
    d4 = np.random.rand(num_directions, 8 * hidden_size).astype(np.float32)

    seq_lens = np.asarray([seq_length] * batch_size).astype(np.int32)

    initial_h = np.random.rand(num_directions, batch_size,
                               hidden_size).astype(np.float32)
    initial_c = np.random.rand(num_directions, batch_size,
                               hidden_size).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        i2 = builder.addInputTensor(d2)
        i3 = builder.addInputTensor(d3)
        i4 = builder.addInputTensor(d4)
        i5 = builder.addInputTensor(seq_lens)
        i6 = builder.addInputTensor(initial_h)
        i7 = builder.addInputTensor(initial_c)
        Y, Y_h, Y_c = builder.lstm([i1, i2, i3, i4, i5, i6, i7])
        builder.addOutputTensor(Y_h)
        return [Y, Y_h, Y_c]

    def reference(ref_data):
        lstm = LSTM_Helper(
            X=d1, W=d2, R=d3, B=d4, initial_h=initial_h, initial_c=initial_c)
        Y, Y_h, Y_c = lstm.step()

        return [Y, Y_h, Y_c]

    op_tester.run(init_builder, reference, 'infer')


def test_slice(op_tester):
    d1 = np.array([[1., 2., 3., 4.], [5., 6., 7., 8.]]).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.slice([i1], [0, 1], [1, 0], [2, 3])
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
        o = builder.slice([i1], [], [1, 0], [2, 3])
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
        o = builder.slice([i1], [0], [-5], [-3])
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
        o = builder.slice([i1], [0, 1], [1, 0], [2, 3])
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
        o = builder.pad([i1], mode, lower_padding + upper_padding, pad_value)
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


def test_gather_rank2_1(op_tester):
    d1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype(np.float32)
    d2 = np.arange(2, dtype=np.int32)
    axis = 0

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        np.random.shuffle(d2)
        i2 = builder.addInputTensor(d2)
        o = builder.gather([i1, i2], axis)
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        out = np.take(d1, d2, axis=axis)
        return [out]

    op_tester.run(init_builder, reference, 'infer')


def test_gather_rank2_2(op_tester):
    d1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype(np.float32)
    d2 = np.arange(2, dtype=np.int32).reshape(1, 2)
    axis = 1

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        np.random.shuffle(d2)
        i2 = builder.addInputTensor(d2)
        o = builder.gather([i1, i2], axis)
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        out = np.take(d1, d2, axis=axis)
        return [out]

    op_tester.run(init_builder, reference, 'infer')


def test_gather_rank3_1(op_tester):
    d1 = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]).astype(np.float32)
    d2 = np.arange(2, dtype=np.int32)
    axis = 2

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        np.random.shuffle(d2)
        i2 = builder.addInputTensor(d2)
        o = builder.gather([i1, i2], axis)
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        out = np.take(d1, d2, axis=axis)
        return [out]

    op_tester.run(init_builder, reference, 'infer')


def test_gather_rank1_1(op_tester):
    d1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]).astype(np.float32)
    d2 = np.arange(2, dtype=np.int32)
    axis = 0

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        np.random.shuffle(d2)
        i2 = builder.addInputTensor(d2)
        o = builder.gather([i1, i2], axis)
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        out = np.take(d1, d2, axis=axis)
        return [out]

    op_tester.run(init_builder, reference, 'infer')


def test_gather_rank1_0(op_tester):
    d1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]).astype(np.float32)
    d2 = np.array([]).astype(np.int32)
    axis = 0

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        np.random.shuffle(d2)
        i2 = builder.addInputTensor(d2)
        o = builder.gather([i1, i2], axis)
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        out = np.take(d1, d2, axis=axis)
        return [out]

    op_tester.run(init_builder, reference, 'infer')


def test_gather_example1(op_tester):
    d1 = np.array([[1.0, 1.2], [2.3, 3.4], [4.5, 5.7]]).astype(np.float32)
    d2 = np.array([[0, 1], [1, 2]]).astype(np.int32)
    axis = 0

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        np.random.shuffle(d2)
        i2 = builder.addInputTensor(d2)
        o = builder.gather([i1, i2], axis)
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        out = np.take(d1, d2, axis=axis)
        return [out]

    op_tester.run(init_builder, reference, 'infer')


def test_gather_example2(op_tester):
    d1 = np.array([[1.0, 1.2, 1.9], [2.3, 3.4, 3.9], [4.5, 5.7,
                                                      5.9]]).astype(np.float32)
    d2 = np.array([[0, 2]]).astype(np.int32)
    axis = 1

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        np.random.shuffle(d2)
        i2 = builder.addInputTensor(d2)
        o = builder.gather([i1, i2], axis)
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        out = np.take(d1, d2, axis=axis)
        return [out]

    op_tester.run(init_builder, reference, 'infer')


# Usage:
#   Add `op_tester` as an argument to a test function
#   In the test function:
#       Create a function to initialize the `Builder`.
#       This function should return a list of anchors.
#
#       Create a function to produce refernece output.
#       This should return a list of refernece values
#       the indices of which should correspond to the
#       anchors they reference.
#       The list of references must be the same length
#       as the list of anchors.
#       To exclude an anchor from testing, use `None`
#       at the anchors index in the refernece list.
#
#       The reference function takes one argument, `ref_data`.
#       `ref_data.getOutputTensorGrad(i)`, will return the gradient
#       of the output tensor at index `i`.
#
#   Call op_tester.run(init_builder, reference)
@pytest.fixture
def op_tester(tmpdir):
    # take a numpy dtype and return a type suitable for TensorInfo
    def convert_dtype(dtype):
        if dtype == np.dtype('float32'):
            return 'FLOAT'

        if dtype == np.dtype('float16'):
            return 'FLOAT16'

        if dtype == np.dtype('int32'):
            return 'INT32'

        raise Exception(
            'bad dtype %s, (only float32 currently supported)' % (dtype, ))

    class Builder:
        def __init__(self):
            self._builder = poponnx.Builder()
            self._input_map = {}
            self._outputs = []

        def addInputTensor(self, data):
            dtype = convert_dtype(data.dtype)
            shape = poponnx.TensorInfo(dtype, data.shape)

            tensor_id = self._builder.addInputTensor(shape)
            self._input_map[tensor_id] = data

            return tensor_id

        def addOutputTensor(self, tensorId):
            self._outputs.append(tensorId)
            self._builder.addOutputTensor(tensorId)

        def __getattr__(self, attr):
            return getattr(self._builder, attr)

    class RefData:
        def __init__(self, outputs, anchor_map):
            self._outputs = outputs
            self._anchor_map = anchor_map

        def getOutputTensorGrad(self, index):
            tensorId = self._outputs[index]
            gradId = 'd__' + tensorId
            return self._anchor_map[gradId]

    class OpTester:
        def __init__(self, logging_dir):
            self.passes = []
            self.logging_dir = logging_dir
            self.rtol = 1e-05
            self.atol = 1e-08
            self.check_shapes = True

        def run(self, init_builder, reference, step_type='infer'):
            assert step_type in ('infer', 'train')

            bld = Builder()

            anchors = {}
            anchorIds = init_builder(bld)
            for anchorId in anchorIds:
                anchors[anchorId] = poponnx.AnchorReturnType("ALL")

            dataFlow = poponnx.DataFlow(1, anchors)

            if (step_type == 'train'):
                optimizer = poponnx.ConstSGD(0.01)
            else:
                optimizer = None

            losses = [poponnx.L1Loss(anchorIds[0], "l1LossVal", 0.1)]
            proto = bld.getModelProto()

            opts = poponnx.SessionOptionsCore()
            opts.logging = {'all': 'TRACE'}
            opts.exportDot = False
            opts.logDir = self.logging_dir

            session = poponnx.Session(
                fnModel=proto,
                dataFeed=dataFlow,
                losses=losses,
                optimizer=optimizer,
                passes=poponnx.Patterns(self.passes),
                userOptions=opts)

            session.setDevice(tu.get_poplar_cpu_device())
            anchor_map = session.initAnchorArrays()

            session.prepareDevice()

            for k, v in bld._input_map.items():
                if not v.flags['C_CONTIGUOUS']:
                    # need to call np.ascontiguousarray
                    # `x = np.ascontiguousarray(x)`
                    raise Exception(
                        'Input "{}" to poponnx.PyStepIO is not C_CONTIGUOS'.
                        format(k))

            stepio = poponnx.PyStepIO(bld._input_map, anchor_map)
            getattr(session, step_type)(stepio)

            ref_out = reference(RefData(bld._outputs, anchor_map))

            def fix_type(t):
                if isinstance(t, torch.Tensor):
                    return t.data.numpy()
                elif isinstance(t, np.ndarray):
                    return t
                elif t is None:
                    return None
                else:
                    raise Exception('unexpected type', type(t))

            ref_out = [fix_type(i) for i in ref_out]
            for index, key in enumerate(anchors):
                if ref_out[index] is not None:
                    print('Testing anchor "{}"...'.format(key))

                    if self.check_shapes:
                        assert anchor_map[key].shape == ref_out[index].shape

                    if (np.allclose(anchor_map[key], ref_out[index], self.rtol,
                                    self.atol) == False):
                        print('rtol:{} atol:{}'.format(self.rtol, self.atol))
                        print('Poponnx : {}', anchor_map[key])
                        print('Torch : {}', ref_out[index])
                        print('{}', np.subtract(anchor_map[key],
                                                ref_out[index]))
                        print(
                            '{}',
                            np.isclose(anchor_map[key], ref_out[index],
                                       self.rtol, self.atol))

                    assert np.allclose(anchor_map[key], ref_out[index],
                                       self.rtol, self.atol)
                else:
                    print('Not Testing anchor "{}" as it is None'.format(key))

    return OpTester(str(tmpdir))
