import numpy as np
import pytest
import poponnx
import torch
from op_tester import op_tester
import test_util as tu


def test_average_pool_1(op_tester):
    d1 = np.random.rand(1, 1, 6, 6).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.averagepool([i1],
                                       kernel_shape=[2, 2],
                                       count_include_pad=0,
                                       pads=[0, 0, 0, 0],
                                       strides=[2, 2])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        avgpool = torch.nn.AvgPool2d(kernel_size=2,
                                     stride=2,
                                     padding=0,
                                     count_include_pad=False)
        out = avgpool(torch.from_numpy(d1))
        return [out]

    op_tester.run(init_builder, reference)


def test_average_pool_2(op_tester):
    d1 = np.random.rand(1, 1, 14, 14).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.averagepool([i1],
                                       kernel_shape=[3, 3],
                                       count_include_pad=0,
                                       pads=[1, 1, 1, 1],
                                       strides=[1, 1])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        avgpool = torch.nn.AvgPool2d(kernel_size=3,
                                     stride=1,
                                     padding=1,
                                     count_include_pad=False)
        out = avgpool(torch.from_numpy(d1))
        return [out]

    op_tester.run(init_builder, reference)


def test_average_pool_3(op_tester):
    d1 = np.random.rand(1, 1, 14, 14).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.averagepool([i1],
                                       kernel_shape=[3, 3],
                                       count_include_pad=0,
                                       pads=[0, 0, 0, 0],
                                       strides=[2, 2])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        avgpool = torch.nn.AvgPool2d(kernel_size=3,
                                     stride=2,
                                     padding=0,
                                     count_include_pad=False)
        out = avgpool(torch.from_numpy(d1))
        return [out]

    op_tester.run(init_builder, reference)


def test_average_pool_with_count_include_pad(op_tester):

    poponnx.getLogger().setLevel("TRACE")

    builder = poponnx.Builder()

    i1 = builder.addInputTensor(poponnx.TensorInfo("FLOAT", [1, 1, 14, 14]))
    o = builder.aiOnnx.averagepool([i1],
                                   kernel_shape=[3, 3],
                                   count_include_pad=1,
                                   pads=[0, 0, 0, 0],
                                   strides=[2, 2])
    builder.addOutputTensor(o)

    builder.removeNodeAttribute("count_include_pad", set([o]))
    builder.addNodeAttribute("count_include_pad", 1, set([o]))

    dataFlow = poponnx.DataFlow(1, {o: poponnx.AnchorReturnType("ALL")})
    optimizer = poponnx.ConstSGD(0.01)
    losses = [poponnx.L1Loss(o, "l1LossVal", 0.1)]
    proto = builder.getModelProto()

    opts = poponnx.SessionOptionsCore()

    with pytest.raises(poponnx.poponnx_exception) as e_info:
        poponnx.TrainingSession(fnModel=proto,
                                dataFeed=dataFlow,
                                losses=losses,
                                optimizer=optimizer,
                                userOptions=opts,
                                deviceInfo=tu.get_poplar_cpu_device())

    assert (e_info.value.args[0].startswith(
        "`count_include_pad` is not supported"))


def test_maxpool_1(op_tester):
    d1 = np.random.rand(1, 1, 16, 16).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        (o, ) = builder.aiOnnx.maxpool([i1],
                                       num_outputs=1,
                                       kernel_shape=[2, 2],
                                       pads=[0, 0, 0, 0],
                                       storage_order=0,
                                       strides=[2, 2])
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
        (o, ) = builder.aiOnnx.maxpool([i1],
                                       num_outputs=1,
                                       kernel_shape=[3, 3],
                                       pads=[1, 1, 1, 1],
                                       storage_order=0,
                                       strides=[1, 1])
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
        (o, ) = builder.aiOnnx.maxpool([i1],
                                       num_outputs=1,
                                       kernel_shape=[5, 5],
                                       pads=[2, 2, 2, 2],
                                       storage_order=0,
                                       strides=[2, 2])
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
        (o, ) = builder.aiOnnx.maxpool([i1],
                                       num_outputs=1,
                                       kernel_shape=[5, 5],
                                       pads=[2, 2, 2, 2],
                                       storage_order=0,
                                       strides=[2, 2])
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
        (o, ) = builder.aiOnnx.maxpool([i1],
                                       num_outputs=1,
                                       kernel_shape=[2, 2],
                                       pads=[0, 0, 0, 0],
                                       storage_order=0,
                                       strides=[2, 2])
        builder.addOutputTensor(o)
        return [
            o,
            poponnx.reservedGradientPrefix() + i1,
            poponnx.reservedGradientPrefix() + o
        ]

    def reference(ref_data):
        t1 = torch.tensor(d1, requires_grad=True)
        avgpool = torch.nn.MaxPool2d(2, 2)
        out = avgpool(t1)
        d__o = ref_data.getOutputTensorGrad(0)
        out.backward(torch.tensor(d__o))
        return [out, t1.grad, None]

    op_tester.passes = ['PreUniRepl']
    op_tester.run(init_builder, reference, step_type='train')


def test_globalmaxpool_2d(op_tester):
    d1 = np.random.rand(1, 1, 6, 6).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.globalmaxpool([i1])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        t1 = torch.tensor(d1, requires_grad=True)
        globalmaxpool = torch.nn.MaxPool2d(6, 6)
        out = globalmaxpool(t1)
        return [out]

    op_tester.passes = ['PreUniRepl']
    op_tester.run(init_builder, reference, step_type='infer')


def test_globalmaxpool_grad_2d(op_tester):
    d1 = np.random.rand(1, 1, 6, 6).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.globalmaxpool([i1])
        builder.addOutputTensor(o)
        return [
            o,
            poponnx.reservedGradientPrefix() + i1,
            poponnx.reservedGradientPrefix() + o
        ]

    def reference(ref_data):
        t1 = torch.tensor(d1, requires_grad=True)
        globalmaxpool = torch.nn.MaxPool2d(6, 6)
        out = globalmaxpool(t1)
        d__o = ref_data.getOutputTensorGrad(0)
        out.backward(torch.tensor(d__o))
        return [out, t1.grad, None]

    op_tester.passes = ['PreUniRepl']
    op_tester.run(init_builder, reference, step_type='train')


# This will fail at the poponnx layer T6966
def test_globalmaxpool_3d(op_tester):
    d1 = np.random.rand(1, 1, 6, 6, 4).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.globalmaxpool([i1])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        t1 = torch.tensor(d1, requires_grad=True)
        globalmaxpool = torch.nn.MaxPool3d(6, 6, 4)
        out = globalmaxpool(t1)
        return [out]

    op_tester.passes = ['PreUniRepl']

    with pytest.raises(poponnx.poplibs_exception) as e_info:
        op_tester.run(init_builder, reference, step_type='infer')

    assert (e_info.value.args[0].startswith("Only 2D pooling supported"))


def test_globalaveragepool_2d(op_tester):
    d1 = np.random.rand(1, 1, 6, 6).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.globalaveragepool([i1])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        t1 = torch.tensor(d1, requires_grad=True)
        globalaveragepool = torch.nn.AvgPool2d(6, 6)
        out = globalaveragepool(t1)
        return [out]

    op_tester.passes = ['PreUniRepl']
    op_tester.run(init_builder, reference, step_type='infer')


def test_globalaveragepool_grad_2d(op_tester):
    d1 = np.random.rand(1, 1, 6, 6).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.globalaveragepool([i1])
        builder.addOutputTensor(o)
        return [
            o,
            poponnx.reservedGradientPrefix() + i1,
            poponnx.reservedGradientPrefix() + o
        ]

    def reference(ref_data):
        t1 = torch.tensor(d1, requires_grad=True)
        globalaveragepool = torch.nn.AvgPool2d(6, 6)
        out = globalaveragepool(t1)
        d__o = ref_data.getOutputTensorGrad(0)
        out.backward(torch.tensor(d__o))
        return [out, t1.grad, None]

    op_tester.passes = ['PreUniRepl']
    op_tester.run(init_builder, reference, step_type='train')
