import numpy as np
import popart
import json
import pytest

# `import test_util` requires adding to sys.path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu


def test_replace_with_identity():
    d1 = np.random.randn(10, 20).astype(np.float32)

    builder = popart.Builder()
    d = builder.addInputTensor("FLOAT", d1.shape)
    o = builder.aiOnnx.transpose([d])
    o = builder.aiOnnx.transpose([o])

    sess = popart.InferenceSession(fnModel=builder.getModelProto(),
                                   deviceInfo=tu.create_test_device(),
                                   dataFlow=popart.DataFlow(1, [o]))
    sess.prepareDevice()

    anchors = sess.initAnchorArrays()

    stepio = popart.PyStepIO({d: d1}, anchors)
    sess.weightsFromHost()

    sess.run(stepio)
    ir = json.loads(sess._serializeIr(popart.IrSerializationFormat.JSON))
    assert len(
        list(filter(lambda op: "Identity" in op["type"], ir["maingraph"]))) > 0
    assert len(
        list(filter(lambda op: "Transpose" in op["type"],
                    ir["maingraph"]))) < 1
    assert np.allclose(anchors[o], d1)


A = 10
B = 20


@pytest.mark.parametrize(
    "reshape", [[1, B, A], [B, 1, A], [B, A, 1], [1, B, 1, 1, A, 1, 1]])
def test_replace_with_reshape(reshape):
    d1 = np.random.randn(A, B).astype(np.float32)

    builder = popart.Builder()
    d = builder.addInputTensor("FLOAT", d1.shape)
    o = builder.aiOnnx.transpose([d], perm=(1, 0))
    o = builder.reshape_const(builder.aiOnnx, [o], reshape)

    perm = list(range(len(reshape)))
    indexA = reshape.index(A)
    indexB = reshape.index(B)
    perm[indexA] = indexB
    perm[indexB] = indexA
    o = builder.aiOnnx.transpose([o], perm=perm)

    sess = popart.InferenceSession(fnModel=builder.getModelProto(),
                                   deviceInfo=tu.create_test_device(),
                                   dataFlow=popart.DataFlow(1, [o]))
    sess.prepareDevice()

    anchors = sess.initAnchorArrays()

    stepio = popart.PyStepIO({d: d1}, anchors)
    sess.weightsFromHost()

    sess.run(stepio)
    ir = json.loads(sess._serializeIr(popart.IrSerializationFormat.JSON))
    assert len(
        list(filter(lambda op: "Reshape" in op["type"], ir["maingraph"]))) > 0
    assert len(
        list(filter(lambda op: "Transpose" in op["type"],
                    ir["maingraph"]))) < 1
    assert np.allclose(anchors[o].flatten(), d1.flatten())


def test_fail_due_to_non_trivial_reshape():
    d1 = np.random.randn(10, 20).astype(np.float32)

    builder = popart.Builder()
    d = builder.addInputTensor("FLOAT", d1.shape)
    o = builder.aiOnnx.transpose([d], perm=(1, 0))
    o = builder.reshape_const(builder.aiOnnx, [o], (1, 5, 40))
    o = builder.aiOnnx.transpose([o], perm=(0, 2, 1))

    sess = popart.InferenceSession(fnModel=builder.getModelProto(),
                                   deviceInfo=tu.create_test_device(),
                                   dataFlow=popart.DataFlow(1, [o]))
    sess.prepareDevice()

    anchors = sess.initAnchorArrays()

    stepio = popart.PyStepIO({d: d1}, anchors)
    sess.weightsFromHost()

    sess.run(stepio)
    ir = json.loads(sess._serializeIr(popart.IrSerializationFormat.JSON))
    assert len(
        list(filter(lambda op: "Reshape" in op["type"], ir["maingraph"]))) == 1
    assert len(
        list(filter(lambda op: "Transpose" in op["type"],
                    ir["maingraph"]))) == 2


def test_fail_due_to_mismatch_permutation():
    d1 = np.random.randn(10, 20, 30).astype(np.float32)

    builder = popart.Builder()
    d = builder.addInputTensor("FLOAT", d1.shape)
    o = builder.aiOnnx.transpose([d], perm=(0, 2, 1))
    o = builder.aiOnnx.transpose([o], perm=(1, 2, 0))

    sess = popart.InferenceSession(fnModel=builder.getModelProto(),
                                   deviceInfo=tu.create_test_device(),
                                   dataFlow=popart.DataFlow(1, [o]))
    sess.prepareDevice()

    anchors = sess.initAnchorArrays()

    stepio = popart.PyStepIO({d: d1}, anchors)
    sess.weightsFromHost()

    sess.run(stepio)
    ir = json.loads(sess._serializeIr(popart.IrSerializationFormat.JSON))
    assert len(
        list(filter(lambda op: "Transpose" in op["type"],
                    ir["maingraph"]))) == 2
