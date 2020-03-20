# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import popart
import sys
import os
import numpy as np
import argparse
import json
import test_util as tu


def conv(b, x, ksize, stride, c_out, name):
    with b.nameScope(name):
        shape_in = b.getTensorShape(x)
        c_in = shape_in[1]

        wshape = [c_out, c_in, ksize, ksize]

        stride = [stride, stride]

        p = int(ksize / 2)
        padding = [p, p, p, p]

        we = b.addInitializedInputTensor(np.zeros(wshape, np.float16),
                                         "weights")
        o = b.aiOnnx.conv([x, we],
                          strides=stride,
                          pads=padding,
                          dilations=[1, 1])

    return o


def maxpool(b, x, ksize, stride, padding):
    shape = b.getTensorShape(x)
    ph = 0
    pl = 0
    if padding == 'same':
        # assume ksize is odd
        pl = int(ksize / 2)
        ph = int(ksize / 2)

    (x, ) = b.aiOnnx.maxpool([x],
                             1,
                             kernel_shape=[ksize, ksize],
                             strides=[stride, stride],
                             pads=[pl, ph, pl, ph])
    return x


def fully_connected(b, x, cout):
    with b.nameScope("fully_connected"):
        # assume 'x' is [N, C, 1, 1]
        shape = b.getTensorShape(x)
        x = b.reshape_const(b.aiOnnx, [x], [shape[0], shape[1]])
        w = b.addInitializedInputTensor(np.zeros([shape[1], cout], np.float16),
                                        "weights")
        o = b.aiOnnx.matmul([x, w])

    return o


def norm(b, x, bshape, training, norm_type):
    if norm_type == "BatchNorm":
        with b.nameScope("bn"):
            scale = b.addInitializedInputTensor(np.zeros(bshape, np.float16),
                                                "scale")
            bias = b.addInitializedInputTensor(np.zeros(bshape, np.float16),
                                               "bias")
            mean = b.addInitializedInputTensor(np.zeros(bshape, np.float16),
                                               "mean")
            var = b.addInitializedInputTensor(np.zeros(bshape, np.float16),
                                              "var")
            if training:
                (x, _, _, _, _) = b.aiOnnx.batchnormalization(
                    [x, scale, bias, mean, var], 5)
            else:
                (x, ) = b.aiOnnx.batchnormalization(
                    [x, scale, bias, mean, var], 1)

    elif norm_type == "GroupNorm":
        # Assume group-size of 32
        with b.nameScope("gn"):
            scale = b.addInitializedInputTensor(np.zeros(bshape, np.float16),
                                                "scale")
            bias = b.addInitializedInputTensor(np.zeros(bshape, np.float16),
                                               "bias")
            (x, _, _) = b.aiGraphcore.groupnormalization([x, scale, bias], 32)

    else:
        raise Exception(
            "Invalid norm type. Must be 'GroupNorm' or 'BatchNorm'")

    return x


def resnet18_block(b, x, first_stride, count, c_int, name, training,
                   norm_type):
    with b.nameScope(name):
        for i in range(count):
            prefix = "L" + str(i)
            sc = x
            shape_in = b.getTensorShape(x)
            stride = (first_stride if (i == 0) else 1)

            x = conv(b, x, 3, stride, c_int, prefix + "/A")
            x = b.aiOnnx.relu([x])
            x = conv(b, x, 3, 1, c_int, prefix + "/B")

            if stride != 1:
                sc = b.aiGraphcore.subsample([sc], [1, 1, stride, stride])

            if shape_in[1] != b.getTensorShape(x)[1]:
                p = b.getTensorShape(x)[1] - shape_in[1]
                sc = b.aiOnnx.pad([sc], pads=[0, 0, 0, 0, 0, p, 0, 0])

            x = b.aiOnnx.add([x, sc])
            x = b.aiOnnx.relu([x])

        x = norm(b, x, c_int, training, norm_type)

    return x


def resnet_data_size():
    return [4, 224, 224]


def get_resnet18_proto(batch_size, training, norm_type):
    # Build model proto
    b = popart.Builder()

    ds = resnet_data_size()
    data_shape = popart.TensorInfo("FLOAT16",
                                   [batch_size, ds[0], ds[1], ds[2]])
    ip = b.addInputTensor(data_shape, "data")

    if training:
        labl_shape = popart.TensorInfo("INT32", [batch_size])
        lb = b.addInputTensor(labl_shape, "labels")
    else:
        lb = None

    x = conv(b, ip, 7, 2, 64, "i64")
    x = b.aiOnnx.relu([x])
    x = maxpool(b, x, 3, 2, 'same')
    x = resnet18_block(b, x, 1, 2, 64, "m64", training, norm_type)
    x = b.aiOnnx.globalaveragepool([x])
    x = fully_connected(b, x, 1000)
    x = b.aiOnnx.softmax([x])
    b.addOutputTensor(x)
    proto = b.getModelProto()

    return proto, ip, lb, x


@tu.requires_ipu_model
def test_mini_resnet_like():
    dirpath = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(dirpath + "/../../graph_util")
    batch_size = 1
    training = True
    norm_type = 'BatchNorm'

    # Get model proto
    proto, ip, lb, op = get_resnet18_proto(batch_size, training, norm_type)

    # Create the onnx session
    opts = popart.SessionOptions()

    session = popart.TrainingSession(
        fnModel=proto,
        dataFeed=popart.DataFlow(1, {"loss": popart.AnchorReturnType("ALL")}),
        optimizer=popart.ConstSGD(0.001),
        losses=[popart.NllLoss(op, lb, "loss")],
        deviceInfo=tu.create_test_device(),
        userOptions=opts)

    session.prepareDevice()

    graph_report = session.getGraphReport()
    graph_report = json.loads(graph_report)

    total_mem = sum(graph_report['memory']['byTile']['total'])
    max_mem = max(graph_report['memory']['byTile']['totalIncludingGaps'])
    print(f'total_mem: {total_mem}')
    print(f'max_mem: {max_mem}')

    # Check that the total memory is within 5% of the reference
    ref_total = 67_730_829
    # If it is more than 5% over, it needs investigating
    assert total_mem / ref_total < 1.05
    # If it is move than 5% under, the reference should probably be updated
    assert total_mem / ref_total > 0.95

    # Check that the maximum memory is within 5% of the reference
    ref_max = 136_472
    # If it is more than 5% over, it needs investigating
    assert max_mem / ref_max < 1.05
    # If it is move than 5% under, the reference should probably be updated
    assert max_mem / ref_max > 0.95
