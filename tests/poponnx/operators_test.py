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
    op_tester.run(init_builder, reference)


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


def test_averagepool(op_tester):
    d1 = np.random.rand(1, 1, 6, 6).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.averagepool([i1], [2, 2], [2, 2], [0, 0, 0, 0])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        avgpool = torch.nn.AvgPool2d(2, 2)
        out = avgpool(torch.from_numpy(d1))
        return [out]

    op_tester.run(init_builder, reference)


def test_maxpool(op_tester):
    d1 = np.random.rand(1, 1, 6, 6).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.maxpool([i1], [2, 2], [2, 2], [0, 0, 0, 0])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        t1 = torch.tensor(d1, requires_grad=True)
        avgpool = torch.nn.MaxPool2d(2, 2)
        out = avgpool(t1)
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

    op_tester.passes = ['PreUniRepl', 'TanhGradOp', 'CoshOp']
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

        def run(self, init_builder, reference, step_type='infer'):
            assert step_type in ('infer', 'train')

            bld = Builder()

            anchors = {}
            anchorIds = init_builder(bld)
            for anchorId in anchorIds:
                anchors[anchorId] = poponnx.AnchorReturnType("ALL")

            dataFlow = poponnx.DataFlow(1, 1, anchors)
            optimizer = poponnx.SGD(0.01)
            losses = [poponnx.L1Loss(anchorIds[0], "l1LossVal", 0.1)]
            proto = bld.getModelProto()

            opts = poponnx.SessionOptionsCore()
            opts.logging = {'all': 'TRACE'}
            opts.exportDot = False

            session = poponnx.Session(
                fnModel=proto,
                dataFeed=dataFlow,
                losses=losses,
                optimizer=optimizer,
                outputdir=self.logging_dir,
                passes=poponnx.Patterns(self.passes),
                userOptions=opts)

            session.setDevice(tu.get_poplar_cpu_device())
            anchor_map = session.initAnchorArrays()

            session.prepareDevice()

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
                    assert np.allclose(anchor_map[key], ref_out[index])

    return OpTester(str(tmpdir))
