# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import numpy as np
import pytest
import popart
import pprint
import json

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
                popart.AnchorReturnType("All"),
                popart.reservedGradientPrefix() + lhs:
                popart.AnchorReturnType("All"),
                popart.reservedGradientPrefix() + rhs:
                popart.AnchorReturnType("All"),
            })

        opts = popart.SessionOptions()
        opts.reportOptions = {"showExecutionSteps": "true"}

        pat = popart.Patterns(popart.PatternsLevel.Default)

        session = popart.TrainingSession(
            fnModel=proto,
            dataFeed=dataFlow,
            userOptions=opts,
            losses=[popart.L1Loss(o, "l1LossVal", 0.1)],
            optimizer=popart.ConstSGD(0.01),
            patterns=pat,
            deviceInfo=tu.create_test_device(opts={"compileIPUCode": False}))

        session.prepareDevice()

        anchors = session.initAnchorArrays()

        inputs = {lhs: lhs_data, rhs: rhs_data, z: zero_data}
        stepio = popart.PyStepIO(inputs, anchors)

        session.run(stepio)

        return anchors[o]

    run_test()


'''
Verify that the following 

    A    B      C    D             A   C        B   D
    |    |      |    |             |   |        |   |
    MAT_MUL     MAT_MUL            CONCAT       CONCAT
       |           |        =>       |            |
   TRANSPOSE       |                 +-- MATMUL---+
       |           |                       | 
       +---MATMUL--+                 +-----+------+
             |                       |            |
                                   SLICE        SLICE
                                     |            |  
                                 TRANSPOSE        |
                                     |            |
                                     + --MATMUL---+
                                           |
 
causes the two MATMULs to be grouped
'''


def test_matmul_grouping_test_1(tmpdir):
    lhs_shape = [6, 32, 32]
    rhs_shape = [6, 32, 64]
    lhs_2_shape = [6, 32, 32]
    rhs_2_shape = [6, 32, 64]
    lhs_data = np.random.rand(6, 32, 32).astype(np.float32)
    rhs_data = np.random.rand(6, 32, 64).astype(np.float32)
    lhs_2_data = np.random.rand(6, 32, 32).astype(np.float32)
    rhs_2_data = np.random.rand(6, 32, 64).astype(np.float32)

    def verify():
        r1 = np.matmul(lhs_data, rhs_data)
        r2 = np.matmul(lhs_2_data, rhs_2_data)
        r2_t = np.transpose(r2, axes=[0, 2, 1])
        return np.matmul(r1, r2_t)

    def run_test(groupingEnabled, verify):
        builder = popart.Builder()

        lhs = builder.addInputTensor(popart.TensorInfo("FLOAT", lhs_shape),
                                     "lhs")
        rhs = builder.addInputTensor(popart.TensorInfo("FLOAT", rhs_shape),
                                     "rhs")
        lhs_2 = builder.addInputTensor(popart.TensorInfo("FLOAT", lhs_2_shape),
                                       "lhs_2")
        rhs_2 = builder.addInputTensor(popart.TensorInfo("FLOAT", rhs_2_shape),
                                       "rhs_2")

        r1 = builder.aiOnnx.matmul([lhs, rhs])

        r2 = builder.aiOnnx.matmul([lhs_2, rhs_2])

        r2_t = builder.aiOnnx.transpose([r2],
                                        perm=[0, 2, 1],
                                        debugPrefix="rhs.transpose")

        o = builder.aiOnnx.matmul([r1, r2_t])

        builder.addOutputTensor(o)

        proto = builder.getModelProto()

        dataFlow = popart.DataFlow(1, {o: popart.AnchorReturnType("All")})

        opts = popart.SessionOptions()
        opts.reportOptions = {"showExecutionSteps": "true"}
        opts.enableOutlining = False
        opts.enableGroupedMatmuls = groupingEnabled

        pat = popart.Patterns(popart.PatternsLevel.Default)

        session = popart.InferenceSession(
            fnModel=proto,
            dataFeed=dataFlow,
            userOptions=opts,
            patterns=pat,
            deviceInfo=tu.create_test_device(opts={"compileIPUCode": False}))

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

        verify(session)

        return anchors[o]

    def verify_no_grouping(session):
        ir = json.loads(session._serializeIr(
            popart.IrSerializationFormat.JSON))

        matmuls = [op for op in ir['maingraph'] if op['type'] == 'MatMul']
        assert (len(matmuls) == 3)

    def verify_grouping(session):
        ir = json.loads(session._serializeIr(
            popart.IrSerializationFormat.JSON))

        matmuls = [op for op in ir['maingraph'] if op['type'] == 'MatMul']
        assert (len(matmuls) == 2)

    assert (np.allclose(run_test(False, verify_no_grouping), verify()))
    assert (np.allclose(run_test(True, verify_grouping), verify()))


'''
Verify that the following 

    A    B       C    D             A   C        B   D
    |    |       |    |
    | TRANSPOSE  |    |
    |    |       |    |             |   |        |   |
    MAT_MUL      MAT_MUL            CONCAT       CONCAT
       |           |        =>       |            |
   TRANSPOSE       |                 +-- MATMUL---+
       |           |                       | 
       +---MATMUL--+                 +-----+------+
             |                       |            |
                                   SLICE        SLICE
                                     |            |  
                                 TRANSPOSE        |
                                     |            |
                                     + --MATMUL---+
                                           |
 
causes the two MATMULs to be grouped
'''


def test_matmul_grouping_test_2(tmpdir):
    A = [1, 32, 64]
    B = [1, 32, 64]
    C = [1, 32, 32]
    D = [1, 32, 64]
    A_data = np.random.rand(1, 32, 64).astype(np.float32)
    B_data = np.random.rand(1, 32, 64).astype(np.float32)
    C_data = np.random.rand(1, 32, 32).astype(np.float32)
    D_data = np.random.rand(1, 32, 64).astype(np.float32)

    def verify():
        b_t = np.transpose(B_data, axes=[0, 2, 1])
        r1 = np.matmul(A_data, b_t)
        r2 = np.matmul(C_data, D_data)
        return np.matmul(r1, r2)

    def run_test(groupingEnabled, verify):
        builder = popart.Builder()

        a = builder.addInputTensor(popart.TensorInfo("FLOAT", A), "A")
        b = builder.addInputTensor(popart.TensorInfo("FLOAT", B), "B")
        c = builder.addInputTensor(popart.TensorInfo("FLOAT", C), "D")
        d = builder.addInputTensor(popart.TensorInfo("FLOAT", D), "D")

        b_t = builder.aiOnnx.transpose([b], perm=[0, 2, 1], debugPrefix="B.T")

        r1 = builder.aiOnnx.matmul([a, b_t], "MATMUL_A")

        r2 = builder.aiOnnx.matmul([c, d], "MATMUL_B")

        o = builder.aiOnnx.matmul([r1, r2], "END")

        builder.addOutputTensor(o)

        proto = builder.getModelProto()

        dataFlow = popart.DataFlow(
            1, {
                o: popart.AnchorReturnType("All"),
                popart.reservedGradientPrefix() + a:
                popart.AnchorReturnType("All"),
                popart.reservedGradientPrefix() + b:
                popart.AnchorReturnType("All"),
                popart.reservedGradientPrefix() + c:
                popart.AnchorReturnType("All"),
                popart.reservedGradientPrefix() + d:
                popart.AnchorReturnType("All")
            })

        opts = popart.SessionOptions()
        opts.reportOptions = {"showExecutionSteps": "true"}
        opts.enableOutlining = False
        opts.enableGroupedMatmuls = groupingEnabled
        opts.dotOpNames = True

        pat = popart.Patterns(popart.PatternsLevel.Default)

        session = popart.TrainingSession(
            fnModel=proto,
            dataFeed=dataFlow,
            userOptions=opts,
            patterns=pat,
            losses=[popart.L1Loss(o, "l1LossVal", 0.1)],
            optimizer=popart.ConstSGD(0.01),
            deviceInfo=tu.create_test_device(opts={"compileIPUCode": False}))

        session.prepareDevice()

        anchors = session.initAnchorArrays()

        inputs = {a: A_data, b: B_data, c: C_data, d: D_data}
        stepio = popart.PyStepIO(inputs, anchors)

        session.run(stepio)

        verify(session)

        return anchors[o]

    def verify_no_grouping(session):
        ir = json.loads(session._serializeIr(
            popart.IrSerializationFormat.JSON))

        matmuls = [op for op in ir['maingraph'] if op['type'] == 'MatMul']
        assert (len(matmuls) == 9)

    def verify_grouping(session):
        ir = json.loads(session._serializeIr(
            popart.IrSerializationFormat.JSON))

        matmuls = [op for op in ir['maingraph'] if op['type'] == 'MatMul']
        assert (len(matmuls) == 7)

    assert (np.allclose(run_test(False, verify_no_grouping), verify()))
    assert (np.allclose(run_test(True, verify_grouping), verify()))


# Verify 2d inputs are expanded to 3d first
def test_matmul_grouping_test_3(tmpdir):
    A = [32, 64]
    B = [32, 64]
    C = [32, 32]
    D = [32, 64]
    A_data = np.random.rand(32, 64).astype(np.float32)
    B_data = np.random.rand(32, 64).astype(np.float32)
    C_data = np.random.rand(32, 32).astype(np.float32)
    D_data = np.random.rand(32, 64).astype(np.float32)

    def verify():
        b_t = np.transpose(B_data, axes=[1, 0])
        r1 = np.matmul(A_data, b_t)
        r2 = np.matmul(C_data, D_data)
        return np.matmul(r1, r2)

    def run_test(groupingEnabled, verify):
        builder = popart.Builder()

        a = builder.addInputTensor(popart.TensorInfo("FLOAT", A), "A")
        b = builder.addInputTensor(popart.TensorInfo("FLOAT", B), "B")
        c = builder.addInputTensor(popart.TensorInfo("FLOAT", C), "D")
        d = builder.addInputTensor(popart.TensorInfo("FLOAT", D), "D")

        b_t = builder.aiOnnx.transpose([b], perm=[1, 0], debugPrefix="B.T")

        r1 = builder.aiOnnx.matmul([a, b_t], "MATMUL_A")

        r2 = builder.aiOnnx.matmul([c, d], "MATMUL_B")

        o = builder.aiOnnx.matmul([r1, r2], "END")

        builder.addOutputTensor(o)

        proto = builder.getModelProto()

        dataFlow = popart.DataFlow(
            1, {
                o: popart.AnchorReturnType("All"),
                popart.reservedGradientPrefix() + a:
                popart.AnchorReturnType("All"),
                popart.reservedGradientPrefix() + b:
                popart.AnchorReturnType("All"),
                popart.reservedGradientPrefix() + c:
                popart.AnchorReturnType("All"),
                popart.reservedGradientPrefix() + d:
                popart.AnchorReturnType("All")
            })

        opts = popart.SessionOptions()
        opts.reportOptions = {"showExecutionSteps": "true"}
        opts.enableOutlining = False
        opts.enableGroupedMatmuls = groupingEnabled
        opts.dotOpNames = True

        pat = popart.Patterns(popart.PatternsLevel.Default)

        session = popart.TrainingSession(
            fnModel=proto,
            dataFeed=dataFlow,
            userOptions=opts,
            patterns=pat,
            losses=[popart.L1Loss(o, "l1LossVal", 0.1)],
            optimizer=popart.ConstSGD(0.01),
            deviceInfo=tu.create_test_device(opts={"compileIPUCode": False}))

        session.prepareDevice()

        anchors = session.initAnchorArrays()

        inputs = {a: A_data, b: B_data, c: C_data, d: D_data}
        stepio = popart.PyStepIO(inputs, anchors)

        session.run(stepio)

        verify(session)

        return anchors[o]

    def verify_no_grouping(session):
        ir = json.loads(session._serializeIr(
            popart.IrSerializationFormat.JSON))

        matmuls = [op for op in ir['maingraph'] if op['type'] == 'MatMul']
        assert (len(matmuls) == 9)

    def verify_grouping(session):
        ir = json.loads(session._serializeIr(
            popart.IrSerializationFormat.JSON))

        matmuls = [op for op in ir['maingraph'] if op['type'] == 'MatMul']
        assert (len(matmuls) == 7)

    assert (np.allclose(run_test(False, verify_no_grouping), verify()))
    assert (np.allclose(run_test(True, verify_grouping), verify()))


# verify that we can group matmuls with different group dimensions
def test_matmul_grouping_test_4(tmpdir):
    A = [2, 3]
    B = [2, 3, 4]
    C = [2, 3]
    D = [2, 3, 4]
    A_data = np.random.rand(2, 3).astype(np.float32)
    B_data = np.random.rand(2, 3, 4).astype(np.float32)
    C_data = np.random.rand(2, 3).astype(np.float32)
    D_data = np.random.rand(2, 3, 4).astype(np.float32)

    def verify():
        r1 = np.matmul(A_data, B_data)
        r2 = np.matmul(C_data, D_data)

        return np.add(r1, r2)

    def verify_grouping():

        A1 = np.reshape(A_data, (1, 1, 2, 3))
        C1 = np.reshape(C_data, (1, 1, 2, 3))

        B1 = np.reshape(B_data, (1, 2, 3, 4))
        D1 = np.reshape(D_data, (1, 2, 3, 4))

        l = np.concatenate((A1, C1), axis=0)
        r = np.concatenate((B1, D1), axis=0)

        o = np.matmul(l, r)

        r1 = o[0:1]
        r2 = o[1:1]

        s1 = np.squeeze(r1)
        s2 = np.squeeze(r2)

        return np.add(s1, s2)

    def run_test(groupingEnabled, verify):
        builder = popart.Builder()

        a = builder.addInputTensor(popart.TensorInfo("FLOAT", A), "A")
        b = builder.addInputTensor(popart.TensorInfo("FLOAT", B), "B")
        c = builder.addInputTensor(popart.TensorInfo("FLOAT", C), "C")
        d = builder.addInputTensor(popart.TensorInfo("FLOAT", D), "D")

        r1 = builder.aiOnnx.matmul([a, b], "MATMUL_A")

        r2 = builder.aiOnnx.matmul([c, d], "MATMUL_B")

        o = builder.aiOnnx.add([r1, r2], "END")

        builder.addOutputTensor(o)

        proto = builder.getModelProto()

        dataFlow = popart.DataFlow(
            1, {
                o: popart.AnchorReturnType("All"),
                popart.reservedGradientPrefix() + a:
                popart.AnchorReturnType("All"),
                popart.reservedGradientPrefix() + b:
                popart.AnchorReturnType("All"),
                popart.reservedGradientPrefix() + c:
                popart.AnchorReturnType("All"),
                popart.reservedGradientPrefix() + d:
                popart.AnchorReturnType("All")
            })

        opts = popart.SessionOptions()
        opts.reportOptions = {"showExecutionSteps": "true"}
        opts.enableOutlining = False
        opts.enableGroupedMatmuls = groupingEnabled
        opts.dotOpNames = True

        pat = popart.Patterns(popart.PatternsLevel.Default)

        session = popart.TrainingSession(
            fnModel=proto,
            dataFeed=dataFlow,
            userOptions=opts,
            patterns=pat,
            losses=[popart.L1Loss(o, "l1LossVal", 0.1)],
            optimizer=popart.ConstSGD(0.01),
            deviceInfo=tu.create_test_device(opts={"compileIPUCode": False}))

        session.prepareDevice()

        anchors = session.initAnchorArrays()

        inputs = {a: A_data, b: B_data, c: C_data, d: D_data}
        stepio = popart.PyStepIO(inputs, anchors)

        session.run(stepio)

        verify(session)

        return anchors[o]

    assert (np.allclose(verify_grouping(), verify()))

    def verify_no_grouping(session):
        ir = json.loads(session._serializeIr(
            popart.IrSerializationFormat.JSON))

        matmuls = [op for op in ir['maingraph'] if op['type'] == 'MatMul']
        assert (len(matmuls) == 6)

    def verify_grouping(session):
        ir = json.loads(session._serializeIr(
            popart.IrSerializationFormat.JSON))

        matmuls = [op for op in ir['maingraph'] if op['type'] == 'MatMul']
        assert (len(matmuls) == 3)

    assert (np.allclose(run_test(False, verify_no_grouping), verify()))
    assert (np.allclose(run_test(True, verify_grouping), verify()))
