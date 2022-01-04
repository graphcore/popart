# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import itertools
import numpy as np
import popart
import torch

# `import test_util` requires adding to sys.path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu


def get_torch_reduction_type(popart_reduction_type):
    if popart_reduction_type == popart.ReductionType.Mean:
        return "mean"

    if popart_reduction_type == popart.ReductionType.Sum:
        return "sum"

    if popart_reduction_type == popart.ReductionType.NoReduction:
        return "none"


def test_l1(op_tester):
    op_tester.rtol = 5e-03
    op_tester.atol = 1e-08

    for dtype, shape in itertools.product(
        [np.float16, np.float32],
        [
            (5, 3, 7),
            (9, 7, 2, 12),
            (3, 256, 256),
            # (2, 1024, 1024) - not working: T9764
        ]):
        data = np.random.random_sample(shape).astype(dtype) - 0.5
        scale = np.random.rand() - 0.5

        print(dtype, shape)

        def init_builder(builder):
            tensor = builder.addInputTensor(data)
            l1 = builder.aiGraphcore.l1loss(
                [tensor], scale, reduction=popart.ReductionType.NoReduction)
            return [l1]

        def reference(ref_data):
            return [np.abs(data) * scale]

        op_tester.passes = ['PreUniRepl']
        op_tester.run(init_builder, reference, 'infer')


def test_l1_training(op_tester):
    op_tester.rtol = 1e-02
    op_tester.atol = 1e-05

    for reduction_type in (popart.ReductionType.Mean,
                           popart.ReductionType.Sum):
        for dtype, shape in itertools.product(
            [np.float16, np.float32],
            [
                (5, 3, 7),
                (9, 7, 2, 12),
                (3, 256, 256),
                #(2, 1024, 1024) - not working: T9764
            ]):
            data = np.random.random_sample(shape).astype(dtype)

            if reduction_type == popart.ReductionType.Mean:
                # Manipulate numbers to avoid signum(0.0) precision
                # difference between FP16 and FP32
                data[np.where(data < 0.5)] -= 1.0
                scale = np.random.rand() - 0.5
            else:
                assert reduction_type == popart.ReductionType.Sum
                if dtype == np.float16:
                    # prevent overflow
                    data = data / shape[-1]
                    data = data.astype(dtype)

            def init_builder(builder):
                result = []
                axes_reduce = []
                tensor = builder.addInputTensor(data)
                l1 = builder.aiGraphcore.l1loss([tensor],
                                                scale,
                                                reduction=reduction_type)
                result.append(l1)
                result.append(popart.reservedGradientPrefix() + tensor)
                return result

            def reference(ref_data):
                result = []

                tensor = torch.tensor(data.astype(np.float32),
                                      requires_grad=True)
                out = torch.nn.L1Loss(
                    reduction=get_torch_reduction_type(reduction_type))(
                        tensor, torch.zeros_like(tensor)) * scale
                result.append(out)
                result.append(tensor)
                out.backward(torch.ones_like(out))
                result[1::2] = [r.grad for r in result[1::2]]
                result = [r.detach().numpy().astype(dtype) for r in result]
                return result

            op_tester.setPatterns(["OpToIdentity"], enableRuntimeAsserts=False)
            op_tester.run(init_builder, reference, 'train')


def test_l1_reduction_equiv(op_tester):
    dshapes = ([2, 3], [2, 4, 4], [5, 1, 3], [1, 1])
    for dshape in dshapes:
        ip_data = np.random.rand(*dshape).astype(np.float32)

        def getAnchors(extraReduction):
            builder = popart.Builder()
            ip = builder.addInitializedInputTensor(ip_data)

            if extraReduction == True:
                l1 = builder.aiGraphcore.l1loss(
                    [ip], 0.1, reduction=popart.ReductionType.NoReduction)
                loss = builder.aiOnnx.reducesum([l1])
            else:
                loss = builder.aiGraphcore.l1loss(
                    [ip], 0.1, reduction=popart.ReductionType.Sum)

            anchors = [loss, popart.reservedGradientPrefix() + ip]

            session = popart.TrainingSession(
                fnModel=builder.getModelProto(),
                loss=loss,
                dataFlow=popart.DataFlow(1, anchors),
                optimizer=popart.ConstSGD(0.1),
                deviceInfo=tu.create_test_device(),
                patterns=popart.Patterns(
                    popart.PatternsLevel.NoPatterns).enableRuntimeAsserts(
                        False))
            session.prepareDevice()
            session.weightsFromHost()
            anchors = session.initAnchorArrays()
            stepio = popart.PyStepIO({}, anchors)
            session.run(stepio)
            return anchors

        # perform sum reduction of individual losses inside l1 op
        lr_anchors = getAnchors(False)

        # perform sum reduction of individual losses outside l1 op
        er_anchors = getAnchors(True)

        # check they are equivalent
        for (id0, a0), (id1, a1) in zip(lr_anchors.items(),
                                        er_anchors.items()):
            assert np.allclose(a0, a1)
