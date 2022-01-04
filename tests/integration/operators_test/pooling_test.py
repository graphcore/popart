# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import numpy as np
import pytest
import popart
import torch
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


def test_average_pool_4(op_tester):
    """
    The Identity case
    """
    d1 = np.random.rand(4, 256, 6, 6).astype(np.float16)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.averagepool([i1],
                                       kernel_shape=[1, 1],
                                       count_include_pad=0,
                                       pads=[0, 0, 0, 0],
                                       strides=[1, 1])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        return [d1]

    op_tester.setPatterns(['OpToIdentity'], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference)


def test_average_pool_fp16(op_tester):
    d1 = np.random.rand(1, 1, 14, 14).astype(np.float16)

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
        out = avgpool(torch.from_numpy(d1.astype(np.float32)))
        return [out.numpy().astype(np.float16)]

    op_tester.atol = 1e-3
    op_tester.rtol = 1e-3
    op_tester.run(init_builder, reference)


def test_average_pool_with_count_include_pad(op_tester):

    popart.getLogger().setLevel("TRACE")

    builder = popart.Builder()

    i1 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1, 1, 14, 14]))
    o = builder.aiOnnx.averagepool([i1],
                                   kernel_shape=[3, 3],
                                   count_include_pad=1,
                                   pads=[0, 0, 0, 0],
                                   strides=[2, 2])
    builder.addOutputTensor(o)

    builder.removeNodeAttribute("count_include_pad", set([o]))
    builder.addNodeAttribute("count_include_pad", 1, set([o]))

    dataFlow = popart.DataFlow(1, {o: popart.AnchorReturnType("All")})
    optimizer = popart.ConstSGD(0.01)
    loss = builder.aiGraphcore.l1loss([o], 0.1)
    proto = builder.getModelProto()

    opts = popart.SessionOptions()

    with pytest.raises(popart.popart_exception) as e_info:
        popart.TrainingSession(fnModel=proto,
                               dataFlow=dataFlow,
                               loss=loss,
                               optimizer=optimizer,
                               userOptions=opts,
                               deviceInfo=tu.create_test_device())

    assert (e_info.value.args[0].startswith(
        "`count_include_pad` is not supported"))


def test_average_pool_invalid_params(op_tester):
    builder = popart.Builder()
    i1 = builder.addInputTensor("FLOAT", [1, 1, 14, 14])
    with pytest.raises(popart.popart_exception) as e_info:
        builder.aiOnnx.averagepool([i1],
                                   kernel_shape=[3, 3],
                                   count_include_pad=1,
                                   pads=[0, 0, 0, 0, 0],
                                   strides=[2, 2])
    assert (e_info.value.args[0].startswith(
        "Padding vector (length 5) does not have"))


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


def test_maxpool_5(op_tester):
    """
    The Identity case
    """
    d1 = np.random.rand(4, 256, 6, 6).astype(np.float16)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        (o, ) = builder.aiOnnx.maxpool([i1],
                                       num_outputs=1,
                                       kernel_shape=[1, 1],
                                       storage_order=0,
                                       pads=[0, 0, 0, 0],
                                       strides=[1, 1])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        return [d1]

    op_tester.setPatterns(['OpToIdentity'], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference)


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
            popart.reservedGradientPrefix() + i1,
            popart.reservedGradientPrefix() + o
        ]

    def reference(ref_data):
        t1 = torch.tensor(d1, requires_grad=True)
        avgpool = torch.nn.MaxPool2d(2, 2)
        out = avgpool(t1)
        d__o = ref_data.getOutputTensorGrad(0)
        out.backward(torch.tensor(d__o))
        return [out, t1.grad, None]

    op_tester.setPatterns(['PreUniRepl', 'OpToIdentity'],
                          enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, step_type='train')


def test_maxpool10_dilations(op_tester):
    d1 = np.random.rand(1, 1, 16, 16).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        (o, ) = builder.aiOnnxOpset10.maxpool([i1],
                                              num_outputs=1,
                                              kernel_shape=[2, 2],
                                              ceil_mode=0,
                                              dilations=[2, 2],
                                              pads=[0, 0, 0, 0],
                                              storage_order=0,
                                              strides=[2, 2])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        t1 = torch.tensor(d1, requires_grad=True)
        avgpool = torch.nn.MaxPool2d(kernel_size=2,
                                     stride=2,
                                     padding=0,
                                     dilation=2)
        out = avgpool(t1)
        return [out]

    with pytest.raises(popart.popart_exception) as e_info:
        op_tester.run(init_builder, reference, step_type='infer')
    assert ("Dilations of value other than 1 are currently not supported." in
            e_info.value.args[0])


def test_maxpool10_ceil_mode(op_tester):
    d1 = np.random.rand(1, 1, 16, 16).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        (o, ) = builder.aiOnnxOpset10.maxpool([i1],
                                              num_outputs=1,
                                              kernel_shape=[3, 3],
                                              ceil_mode=1,
                                              pads=[0, 0, 0, 0],
                                              storage_order=0,
                                              strides=[2, 2])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        t1 = torch.tensor(d1, requires_grad=True)
        avgpool = torch.nn.MaxPool2d(kernel_size=3,
                                     stride=2,
                                     padding=0,
                                     ceil_mode=True)
        out = avgpool(t1)
        print(out)
        return [out]

    op_tester.run(init_builder, reference, step_type='infer')


def test_maxpool10_ceil_mode_grad(op_tester):
    d1 = np.random.rand(1, 1, 16, 16).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        (o, ) = builder.aiOnnxOpset10.maxpool([i1],
                                              num_outputs=1,
                                              kernel_shape=[3, 3],
                                              ceil_mode=1,
                                              pads=[0, 0, 0, 0],
                                              storage_order=0,
                                              strides=[2, 2])
        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + i1,
            popart.reservedGradientPrefix() + o
        ]

    def init_builder_manual_padding(builder):
        i1 = builder.addInputTensor(d1)
        (o, ) = builder.aiOnnxOpset10.maxpool([i1],
                                              num_outputs=1,
                                              kernel_shape=[3, 3],
                                              ceil_mode=0,
                                              pads=[0, 0, 1, 1],
                                              storage_order=0,
                                              strides=[2, 2])
        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + i1,
            popart.reservedGradientPrefix() + o
        ]

    def reference(ref_data):
        t1 = torch.tensor(d1, requires_grad=True)
        avgpool = torch.nn.MaxPool2d(kernel_size=3,
                                     stride=2,
                                     padding=0,
                                     ceil_mode=True)
        out = avgpool(t1)
        d__o = ref_data.getOutputTensorGrad(0)
        out.backward(torch.tensor(d__o))
        return [out, t1.grad, None]

    op_tester.run(init_builder, reference, step_type='train')
    op_tester.run(init_builder_manual_padding, reference, step_type='train')


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

    op_tester.setPatterns(['PreUniRepl'], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, step_type='infer')


def test_globalmaxpool_grad_2d(op_tester):
    d1 = np.random.rand(1, 1, 6, 6).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.globalmaxpool([i1])
        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + i1,
            popart.reservedGradientPrefix() + o
        ]

    def reference(ref_data):
        t1 = torch.tensor(d1, requires_grad=True)
        globalmaxpool = torch.nn.MaxPool2d(6, 6)
        out = globalmaxpool(t1)
        d__o = ref_data.getOutputTensorGrad(0)
        out.backward(torch.tensor(d__o))
        return [out, t1.grad, None]

    op_tester.setPatterns(['PreUniRepl', 'OpToIdentity'],
                          enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, step_type='train')


# This will fail at the popart layer T6966
def test_globalmaxpool_3d(op_tester):
    d1 = np.random.rand(1, 1, 6, 6, 4).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.globalmaxpool([i1])
        builder.addOutputTensor(o)
        return [o]

    def reference(ref_data):
        t1 = torch.tensor(d1, requires_grad=True)
        globalmaxpool = torch.nn.MaxPool3d((6, 6, 4))
        out = globalmaxpool(t1)
        return [out]

    op_tester.setPatterns(['PreUniRepl'], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, step_type='infer')


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

    op_tester.setPatterns(['PreUniRepl'], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, step_type='infer')


def test_globalaveragepool_grad_2d(op_tester):
    d1 = np.random.rand(1, 1, 6, 6).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(d1)
        o = builder.aiOnnx.globalaveragepool([i1])
        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + i1,
            popart.reservedGradientPrefix() + o
        ]

    def reference(ref_data):
        t1 = torch.tensor(d1, requires_grad=True)
        globalaveragepool = torch.nn.AvgPool2d(6, 6)
        out = globalaveragepool(t1)
        d__o = ref_data.getOutputTensorGrad(0)
        out.backward(torch.tensor(d__o))
        return [out, t1.grad, None]

    op_tester.setPatterns(['PreUniRepl', 'OpToIdentity'],
                          enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, step_type='train')
