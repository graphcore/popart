import numpy as np
import pytest
import popart
import pprint

# `import test_util` requires adding to sys.path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu


def test_matmul_1d(tmpdir):
    lhs_shape = [4]
    rhs_shape = [4]
    lhs_data = np.random.rand(4).astype(np.float32)
    rhs_data = np.random.rand(4).astype(np.float32)

    zero_data = np.zeros(2).astype(np.float32)

    def run_test():
        builder = popart.Builder()

        lhs = builder.addInputTensor(popart.TensorInfo("FLOAT", lhs_shape),
                                     "lhs")
        rhs = builder.addInputTensor(popart.TensorInfo("FLOAT", rhs_shape),
                                     "rhs")

        z = builder.addInputTensor(popart.TensorInfo("FLOAT", [2]), "zero")

        t1 = builder.aiOnnx.matmul([lhs, rhs])

        o = builder.aiOnnx.add([z, t1])

        builder.addOutputTensor(o)

        proto = builder.getModelProto()

        dataFlow = popart.DataFlow(
            1, {
                o:
                popart.AnchorReturnType("ALL"),
                popart.reservedGradientPrefix() + lhs:
                popart.AnchorReturnType("ALL"),
                popart.reservedGradientPrefix() + rhs:
                popart.AnchorReturnType("ALL"),
            })

        opts = popart.SessionOptions()
        opts.reportOptions = {"showExecutionSteps": "true"}

        pat = popart.Patterns(popart.PatternsLevel.DEFAULT)
        pat.MatMulLhsGradOp = False
        pat.MatMulRhsGradOp = False

        session = popart.TrainingSession(
            fnModel=proto,
            dataFeed=dataFlow,
            userOptions=opts,
            losses=[popart.L1Loss(o, "l1LossVal", 0.1)],
            optimizer=popart.ConstSGD(0.01),
            passes=pat,
            deviceInfo=tu.get_ipu_model(compileIPUCode=False))

        session.prepareDevice()

        anchors = session.initAnchorArrays()

        inputs = {lhs: lhs_data, rhs: rhs_data, z: zero_data}
        stepio = popart.PyStepIO(inputs, anchors)

        session.run(stepio)

        return anchors[o]

    run_test()


def test_matmul_grouping(tmpdir):
    lhs_shape = [8, 6, 32, 64]
    rhs_shape = [8, 6, 32, 64]
    lhs_2_shape = [8, 6, 32, 32]
    rhs_2_shape = [8, 6, 32, 64]
    lhs_data = np.random.rand(8, 6, 32, 64).astype(np.float32)
    rhs_data = np.random.rand(8, 6, 32, 64).astype(np.float32)
    lhs_2_data = np.random.rand(8, 6, 32, 32).astype(np.float32)
    rhs_2_data = np.random.rand(8, 6, 32, 64).astype(np.float32)

    def run_test():
        builder = popart.Builder()

        lhs = builder.addInputTensor(popart.TensorInfo("FLOAT", lhs_shape),
                                     "lhs")
        rhs = builder.addInputTensor(popart.TensorInfo("FLOAT", rhs_shape),
                                     "rhs")
        lhs_2 = builder.addInputTensor(popart.TensorInfo("FLOAT", lhs_2_shape),
                                       "lhs_2")
        rhs_2 = builder.addInputTensor(popart.TensorInfo("FLOAT", rhs_2_shape),
                                       "rhs_2")

        rhs_t = builder.aiOnnx.transpose([rhs],
                                         perm=[0, 1, 3, 2],
                                         debugPrefix="rhs.transpose")
        r1 = builder.aiOnnx.matmul([lhs, rhs_t])

        r2 = builder.aiOnnx.matmul([lhs_2, rhs_2])

        o = builder.aiOnnx.matmul([r1, r2])

        builder.addOutputTensor(o)

        proto = builder.getModelProto()

        dataFlow = popart.DataFlow(
            1, {
                o:
                popart.AnchorReturnType("ALL"),
                popart.reservedGradientPrefix() + lhs:
                popart.AnchorReturnType("ALL"),
                popart.reservedGradientPrefix() + rhs:
                popart.AnchorReturnType("ALL"),
                popart.reservedGradientPrefix() + lhs_2:
                popart.AnchorReturnType("ALL"),
                popart.reservedGradientPrefix() + rhs_2:
                popart.AnchorReturnType("ALL")
            })

        opts = popart.SessionOptions()
        opts.reportOptions = {"showExecutionSteps": "true"}

        pat = popart.Patterns(popart.PatternsLevel.DEFAULT)
        pat.MatMulLhsGradOp = True
        pat.MatMulRhsGradOp = True

        session = popart.TrainingSession(
            fnModel=proto,
            dataFeed=dataFlow,
            userOptions=opts,
            losses=[popart.L1Loss(o, "l1LossVal", 0.1)],
            optimizer=popart.ConstSGD(0.01),
            passes=pat,
            deviceInfo=tu.get_ipu_model(compileIPUCode=False))

        session.prepareDevice()

        anchors = session.initAnchorArrays()

        inputs = {
            lhs: lhs_data,
            rhs: rhs_data,
            lhs_2: lhs_2_data,
            rhs_2: rhs_2_data
        }
        stepio = popart.PyStepIO(inputs, anchors)

        session.run(stepio)

        return anchors[o]

    run_test()

    # Need to check the number of MatMuls & Make sure that the MatMulXXXGradOps have been removed
