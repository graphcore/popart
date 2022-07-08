# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
import torch


def test_cumsum_1d(op_tester):
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0]).astype(np.float32)
    axis = np.array(0).astype(np.int32)

    def init_builder(builder):
        i0 = builder.addInputTensor(x)
        i1 = builder.aiOnnxOpset11.constant(axis)
        o = builder.aiOnnxOpset11.cumsum([i0, i1])
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        tx = torch.tensor(x)
        out = torch.cumsum(tx, axis.item(0))
        return [out]

    op_tester.run(init_builder, reference, "infer")


def test_cumsum_1d_exclusive(op_tester):
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0]).astype(np.float32)
    axis = np.array(0).astype(np.int32)
    expected = np.array([0.0, 1.0, 3.0, 6.0, 10.0]).astype(np.float32)

    def init_builder(builder):
        i0 = builder.addInputTensor(x)
        i1 = builder.aiOnnxOpset11.constant(axis)
        o = builder.aiOnnxOpset11.cumsum([i0, i1], exclusive=1)
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        out = torch.tensor(expected)
        return [out]

    op_tester.run(init_builder, reference, "infer")


def test_cumsum_1d_reverse(op_tester):
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0]).astype(np.float32)
    axis = np.array(0).astype(np.int32)

    def init_builder(builder):
        i0 = builder.addInputTensor(x)
        i1 = builder.aiOnnxOpset11.constant(axis)
        o = builder.aiOnnxOpset11.cumsum([i0, i1], reverse=1)
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        tx = torch.tensor(x)
        tx = torch.flip(tx, [0])
        out = torch.cumsum(tx, 0)
        out = torch.flip(out, [0])
        return [out]

    op_tester.run(init_builder, reference, "infer")


def test_cumsum_1d_reverse_exclusive(op_tester):
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0]).astype(np.float32)
    axis = np.array(0).astype(np.int32)
    expected = np.array([14.0, 12.0, 9.0, 5.0, 0.0]).astype(np.float32)

    def init_builder(builder):
        i0 = builder.addInputTensor(x)
        i1 = builder.aiOnnxOpset11.constant(axis)
        o = builder.aiOnnxOpset11.cumsum([i0, i1], reverse=1, exclusive=1)
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        out = torch.tensor(expected)
        return [out]

    op_tester.run(init_builder, reference, "infer")


def test_cumsum_2d_axis_0(op_tester):
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).astype(np.float32).reshape((2, 3))
    axis = np.array(0).astype(np.int32)

    def init_builder(builder):
        i0 = builder.addInputTensor(x)
        i1 = builder.aiOnnxOpset11.constant(axis)
        o = builder.aiOnnxOpset11.cumsum([i0, i1])
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        tx = torch.tensor(x)
        out = torch.cumsum(tx, axis.item(0))
        return [out]

    op_tester.run(init_builder, reference, "infer")


def test_cumsum_2d_axis_1(op_tester):
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).astype(np.float32).reshape((2, 3))
    axis = np.array(1).astype(np.int32)

    def init_builder(builder):
        i0 = builder.addInputTensor(x)
        i1 = builder.aiOnnxOpset11.constant(axis)
        o = builder.aiOnnxOpset11.cumsum([i0, i1])
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        tx = torch.tensor(x)
        out = torch.cumsum(tx, axis.item(0))
        return [out]

    op_tester.run(init_builder, reference, "infer")


def test_cumsum_2d_negative_axis(op_tester):
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).astype(np.float32).reshape((2, 3))
    axis = np.array(-1).astype(np.int32)

    def init_builder(builder):
        i0 = builder.addInputTensor(x)
        i1 = builder.aiOnnxOpset11.constant(axis)
        o = builder.aiOnnxOpset11.cumsum([i0, i1])
        builder.addOutputTensor(o)
        return [o]

    def reference(_):  # ref_data is an unused argument
        tx = torch.tensor(x)
        out = torch.cumsum(tx, axis.item(0))
        return [out]

    op_tester.run(init_builder, reference, "infer")


def test_cumsum_3d(op_tester):
    a0 = np.array(
        [
            [
                [0, 1, 2, 3, 4],
                [5, 6, 7, 8, 9],
                [10, 11, 12, 13, 14],
                [15, 16, 17, 18, 19],
            ],
            [
                [20, 22, 24, 26, 28],
                [30, 32, 34, 36, 38],
                [40, 42, 44, 46, 48],
                [50, 52, 54, 56, 58],
            ],
            [
                [60, 63, 66, 69, 72],
                [75, 78, 81, 84, 87],
                [90, 93, 96, 99, 102],
                [105, 108, 111, 114, 117],
            ],
        ]
    ).astype(np.float32)

    a1 = np.array(
        [
            [
                [0, 1, 2, 3, 4],
                [5, 7, 9, 11, 13],
                [15, 18, 21, 24, 27],
                [30, 34, 38, 42, 46],
            ],
            [
                [20, 21, 22, 23, 24],
                [45, 47, 49, 51, 53],
                [75, 78, 81, 84, 87],
                [110, 114, 118, 122, 126],
            ],
            [
                [40, 41, 42, 43, 44],
                [85, 87, 89, 91, 93],
                [135, 138, 141, 144, 147],
                [190, 194, 198, 202, 206],
            ],
        ]
    ).astype(np.float32)

    a2 = np.array(
        [
            [
                [0, 1, 3, 6, 10],
                [5, 11, 18, 26, 35],
                [10, 21, 33, 46, 60],
                [15, 31, 48, 66, 85],
            ],
            [
                [20, 41, 63, 86, 110],
                [25, 51, 78, 106, 135],
                [30, 61, 93, 126, 160],
                [35, 71, 108, 146, 185],
            ],
            [
                [40, 81, 123, 166, 210],
                [45, 91, 138, 186, 235],
                [50, 101, 153, 206, 260],
                [55, 111, 168, 226, 285],
            ],
        ]
    ).astype(np.float32)
    am1 = a2
    am2 = a1
    am3 = a0
    expected = {-3: am3, -2: am2, -1: am1, 0: a0, 1: a1, 2: a2}

    testAxis = np.array([-3, -2, -1, 0, 1, 2]).astype(np.int32)
    for a in testAxis:
        x = np.arange(60).astype(np.float32).reshape((3, 4, 5))
        axis = np.array(a).astype(np.int32)

        def init_builder(builder):
            i0 = builder.addInputTensor(x)
            i1 = builder.aiOnnxOpset11.constant(axis)
            o = builder.aiOnnxOpset11.cumsum([i0, i1])
            builder.addOutputTensor(o)
            return [o]

        def reference(_):  # ref_data is an unused argument
            out = torch.tensor(expected[a])
            return [out]

        op_tester.run(init_builder, reference, "infer")


def test_cumsum_3d_v2(op_tester):
    testAxis = [-3, -2, -1, 0, 1, 2]
    for a in testAxis:
        x = np.arange(60).astype(np.float32).reshape((3, 4, 5))
        axis = np.array(a).astype(np.int32)

        def init_builder(builder):
            i0 = builder.addInputTensor(x)
            i1 = builder.aiOnnxOpset11.constant(axis)
            o = builder.aiOnnxOpset11.cumsum([i0, i1])
            builder.addOutputTensor(o)
            return [o]

        def reference(_):  # ref_data is an unused argument
            tx = torch.tensor(x)
            out = torch.cumsum(tx, a)
            return [out]

        op_tester.run(init_builder, reference, "infer")


def test_cumsum_3d_reverse(op_tester):
    a0 = np.array(
        [
            [
                [60, 63, 66, 69, 72],
                [75, 78, 81, 84, 87],
                [90, 93, 96, 99, 102],
                [105, 108, 111, 114, 117],
            ],
            [
                [60, 62, 64, 66, 68],
                [70, 72, 74, 76, 78],
                [80, 82, 84, 86, 88],
                [90, 92, 94, 96, 98],
            ],
            [
                [40, 41, 42, 43, 44],
                [45, 46, 47, 48, 49],
                [50, 51, 52, 53, 54],
                [55, 56, 57, 58, 59],
            ],
        ]
    ).astype(np.float32)

    a1 = np.array(
        [
            [
                [30, 34, 38, 42, 46],
                [30, 33, 36, 39, 42],
                [25, 27, 29, 31, 33],
                [15, 16, 17, 18, 19],
            ],
            [
                [110, 114, 118, 122, 126],
                [90, 93, 96, 99, 102],
                [65, 67, 69, 71, 73],
                [35, 36, 37, 38, 39],
            ],
            [
                [190, 194, 198, 202, 206],
                [150, 153, 156, 159, 162],
                [105, 107, 109, 111, 113],
                [55, 56, 57, 58, 59],
            ],
        ]
    ).astype(np.float32)

    a2 = np.array(
        [
            [
                [10, 10, 9, 7, 4],
                [35, 30, 24, 17, 9],
                [60, 50, 39, 27, 14],
                [85, 70, 54, 37, 19],
            ],
            [
                [110, 90, 69, 47, 24],
                [135, 110, 84, 57, 29],
                [160, 130, 99, 67, 34],
                [185, 150, 114, 77, 39],
            ],
            [
                [210, 170, 129, 87, 44],
                [235, 190, 144, 97, 49],
                [260, 210, 159, 107, 54],
                [285, 230, 174, 117, 59],
            ],
        ]
    ).astype(np.float32)
    am1 = a2
    am2 = a1
    am3 = a0
    expected = {-3: am3, -2: am2, -1: am1, 0: a0, 1: a1, 2: a2}

    testAxis = np.array([-3, -2, -1, 0, 1, 2]).astype(np.int32)
    for a in testAxis:
        x = np.arange(60).astype(np.float32).reshape((3, 4, 5))
        axis = np.array(a).astype(np.int32)

        def init_builder(builder):
            i0 = builder.addInputTensor(x)
            i1 = builder.aiOnnxOpset11.constant(axis)
            o = builder.aiOnnxOpset11.cumsum([i0, i1], reverse=1)
            builder.addOutputTensor(o)
            return [o]

        def reference(_):  # ref_data is an unused argument
            out = torch.tensor(expected[a])
            return [out]

        op_tester.run(init_builder, reference, "infer")


def test_cumsum_3d_reverse_v2(op_tester):
    testAxis = [-3, -2, -1, 0, 1, 2]
    for a in testAxis:
        x = np.arange(60).astype(np.float32).reshape((3, 4, 5))
        axis = np.array(a).astype(np.int32)

        def init_builder(builder):
            i0 = builder.addInputTensor(x)
            i1 = builder.aiOnnxOpset11.constant(axis)
            o = builder.aiOnnxOpset11.cumsum([i0, i1], reverse=1)
            builder.addOutputTensor(o)
            return [o]

        def reference(_):  # ref_data is an unused argument
            tx = torch.tensor(x)
            tx = torch.flip(tx, [a])
            out = torch.cumsum(tx, a)
            out = torch.flip(out, [a])
            return [out]

        op_tester.run(init_builder, reference, "infer")


def test_cumsum_3d_exclusive(op_tester):
    # Expected from tf as pytorch does not support
    # exclusive and reverse.
    a0 = np.array(
        [
            [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            [
                [0, 1, 2, 3, 4],
                [5, 6, 7, 8, 9],
                [10, 11, 12, 13, 14],
                [15, 16, 17, 18, 19],
            ],
            [
                [20, 22, 24, 26, 28],
                [30, 32, 34, 36, 38],
                [40, 42, 44, 46, 48],
                [50, 52, 54, 56, 58],
            ],
        ]
    ).astype(np.float32)

    a1 = np.array(
        [
            [[0, 0, 0, 0, 0], [0, 1, 2, 3, 4], [5, 7, 9, 11, 13], [15, 18, 21, 24, 27]],
            [
                [0, 0, 0, 0, 0],
                [20, 21, 22, 23, 24],
                [45, 47, 49, 51, 53],
                [75, 78, 81, 84, 87],
            ],
            [
                [0, 0, 0, 0, 0],
                [40, 41, 42, 43, 44],
                [85, 87, 89, 91, 93],
                [135, 138, 141, 144, 147],
            ],
        ]
    ).astype(np.float32)

    a2 = np.array(
        [
            [
                [0, 0, 1, 3, 6],
                [0, 5, 11, 18, 26],
                [0, 10, 21, 33, 46],
                [0, 15, 31, 48, 66],
            ],
            [
                [0, 20, 41, 63, 86],
                [0, 25, 51, 78, 106],
                [0, 30, 61, 93, 126],
                [0, 35, 71, 108, 146],
            ],
            [
                [0, 40, 81, 123, 166],
                [0, 45, 91, 138, 186],
                [0, 50, 101, 153, 206],
                [0, 55, 111, 168, 226],
            ],
        ]
    ).astype(np.float32)
    am1 = a2
    am2 = a1
    am3 = a0
    expected = {-3: am3, -2: am2, -1: am1, 0: a0, 1: a1, 2: a2}

    testAxis = np.array([-3, -2, -1, 0, 1, 2]).astype(np.int32)
    for a in testAxis:
        x = np.arange(60).astype(np.float32).reshape((3, 4, 5))
        axis = np.array(a).astype(np.int32)

        def init_builder(builder):
            i0 = builder.addInputTensor(x)
            i1 = builder.aiOnnxOpset11.constant(axis)
            o = builder.aiOnnxOpset11.cumsum([i0, i1], exclusive=1)
            builder.addOutputTensor(o)
            return [o]

        def reference(_):  # ref_data is an unused argument
            out = torch.tensor(expected[a])
            return [out]

        op_tester.run(init_builder, reference, "infer")


def test_cumsum_grad_1d(op_tester):
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0]).astype(np.float32)
    axis = np.array(0).astype(np.int32)

    def init_builder(builder):
        i0 = builder.addInputTensor(x)
        i1 = builder.aiOnnxOpset11.constant(axis)
        o = builder.aiOnnxOpset11.cumsum([i0, i1])
        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + i0,
            popart.reservedGradientPrefix() + o,
        ]

    def reference(ref_data):
        tx = torch.tensor(x, requires_grad=True)
        out = torch.cumsum(tx, axis.item(0))
        d__o = ref_data.getOutputTensorGrad(0)
        out.backward(torch.tensor(d__o))
        return [out, tx.grad, None]

    op_tester.run(init_builder, reference, "train")


def test_cumsum_grad_1d_reverse(op_tester):
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0]).astype(np.float32)
    axis = np.array(0).astype(np.int32)

    def init_builder(builder):
        i0 = builder.addInputTensor(x)
        i1 = builder.aiOnnxOpset11.constant(axis)
        o = builder.aiOnnxOpset11.cumsum([i0, i1], reverse=1)
        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + i0,
            popart.reservedGradientPrefix() + o,
        ]

    def reference(ref_data):
        tx = torch.tensor(x, requires_grad=True)
        tx = torch.flip(tx, [0])
        out = torch.cumsum(tx, 0)
        out = torch.flip(out, [0])
        d__o = ref_data.getOutputTensorGrad(0)
        out.backward(torch.tensor(d__o))
        return [out, tx.grad, None]

    op_tester.run(init_builder, reference, "train")


def test_cumsum_grad_2d_axis_0(op_tester):
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).astype(np.float32).reshape((2, 3))
    axis = np.array(0).astype(np.int32)

    def init_builder(builder):
        i0 = builder.addInputTensor(x)
        i1 = builder.aiOnnxOpset11.constant(axis)
        o = builder.aiOnnxOpset11.cumsum([i0, i1])
        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + i0,
            popart.reservedGradientPrefix() + o,
        ]

    def reference(ref_data):
        tx = torch.tensor(x, requires_grad=True)
        out = torch.cumsum(tx, axis.item(0))
        d__o = ref_data.getOutputTensorGrad(0)
        out.backward(torch.tensor(d__o))
        return [out, tx.grad, None]

    op_tester.run(init_builder, reference, "train")


def test_cumsum_grad_2d_axis_1(op_tester):
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).astype(np.float32).reshape((2, 3))
    axis = np.array(1).astype(np.int32)

    def init_builder(builder):
        i0 = builder.addInputTensor(x)
        i1 = builder.aiOnnxOpset11.constant(axis)
        o = builder.aiOnnxOpset11.cumsum([i0, i1])
        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + i0,
            popart.reservedGradientPrefix() + o,
        ]

    def reference(ref_data):
        tx = torch.tensor(x, requires_grad=True)
        out = torch.cumsum(tx, axis.item(0))
        d__o = ref_data.getOutputTensorGrad(0)
        out.backward(torch.tensor(d__o))
        return [out, tx.grad, None]

    op_tester.run(init_builder, reference, "train")


def test_cumsum_grad_2d_negative_axis(op_tester):
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).astype(np.float32).reshape((2, 3))
    axis = np.array(-1).astype(np.int32)

    def init_builder(builder):
        i0 = builder.addInputTensor(x)
        i1 = builder.aiOnnxOpset11.constant(axis)
        o = builder.aiOnnxOpset11.cumsum([i0, i1])
        builder.addOutputTensor(o)
        return [
            o,
            popart.reservedGradientPrefix() + i0,
            popart.reservedGradientPrefix() + o,
        ]

    def reference(ref_data):
        tx = torch.tensor(x, requires_grad=True)
        out = torch.cumsum(tx, axis.item(0))
        d__o = ref_data.getOutputTensorGrad(0)
        out.backward(torch.tensor(d__o))
        return [out, tx.grad, None]

    op_tester.run(init_builder, reference, "train")


def test_cumsum_grad_3d(op_tester):
    testAxis = [-3, -2, -1, 0, 1, 2]
    for a in testAxis:
        x = np.arange(60).astype(np.float32).reshape((3, 4, 5))
        axis = np.array(a).astype(np.int32)

        def init_builder(builder):
            i0 = builder.addInputTensor(x)
            i1 = builder.aiOnnxOpset11.constant(axis)
            o = builder.aiOnnxOpset11.cumsum([i0, i1])
            builder.addOutputTensor(o)
            return [
                o,
                popart.reservedGradientPrefix() + i0,
                popart.reservedGradientPrefix() + o,
            ]

        def reference(ref_data):
            tx = torch.tensor(x, requires_grad=True)
            out = torch.cumsum(tx, a)
            d__o = ref_data.getOutputTensorGrad(0)
            out.backward(torch.tensor(d__o))
            return [out, tx.grad, None]

        op_tester.run(init_builder, reference, "train")


def test_cumsum_grad_3d_reverse(op_tester):
    testAxis = [-3, -2, -1, 0, 1, 2]
    for a in testAxis:
        x = np.arange(60).astype(np.float32).reshape((3, 4, 5))
        axis = np.array(a).astype(np.int32)

        def init_builder(builder):
            i0 = builder.addInputTensor(x)
            i1 = builder.aiOnnxOpset11.constant(axis)
            o = builder.aiOnnxOpset11.cumsum([i0, i1], reverse=1)
            builder.addOutputTensor(o)
            return [
                o,
                popart.reservedGradientPrefix() + i0,
                popart.reservedGradientPrefix() + o,
            ]

        def reference(ref_data):
            tx = torch.tensor(x, requires_grad=True)
            tx = torch.flip(tx, [a])
            out = torch.cumsum(tx, a)
            out = torch.flip(out, [a])
            d__o = ref_data.getOutputTensorGrad(0)
            out.backward(torch.tensor(d__o))
            return [out, tx.grad, None]

        op_tester.run(init_builder, reference, "train")
