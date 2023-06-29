# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
import sys

# `import test_util` requires adding to sys.path
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu

import ctypes

ctypes.cdll.LoadLibrary("./libattr_pylist_float_customOp.so")


def test_customOp_python_floats():
    builder = popart.Builder()

    input_tensor = builder.addInputTensor(popart.TensorInfo("FLOAT", [1]))

    output_tensor = builder.customOp(
        opName="CustomopAttrPylist",
        opVersion=1,
        inputs=[input_tensor],
        domain="com.acme",
        attributes={"values": [1.0, 2.0, 3.0]},
    )[0]

    builder.addOutputTensor(output_tensor)
    proto = builder.getModelProto()

    # Create a runtime environment
    anchors = {output_tensor: popart.AnchorReturnType("All")}
    dataFlow = popart.DataFlow(1, anchors)

    with tu.create_test_device() as device:
        session = popart.InferenceSession(proto, dataFlow, device)

        session.prepareDevice()

        anchors = session.initAnchorArrays()

        input = np.random.rand(1).astype(np.float32)

        stepio = popart.PyStepIO({input_tensor: input}, anchors)

        session.run(stepio)
        assert np.allclose(input, anchors["CustomopAttrPylist:0"])


def test_customOp_python_ints():
    builder = popart.Builder()

    input_tensor = builder.addInputTensor(popart.TensorInfo("FLOAT", [1]))

    output_tensor = builder.customOp(
        opName="CustomopAttrPylist",
        opVersion=1,
        inputs=[input_tensor],
        domain="com.acme",
        attributes={"values": [1, 2, 3]},
    )[0]

    builder.addOutputTensor(output_tensor)
    proto = builder.getModelProto()

    # Create a runtime environment
    anchors = {output_tensor: popart.AnchorReturnType("All")}
    dataFlow = popart.DataFlow(1, anchors)

    with tu.create_test_device() as device:
        session = popart.InferenceSession(proto, dataFlow, device)

        session.prepareDevice()

        anchors = session.initAnchorArrays()

        input = np.random.rand(1).astype(np.float32)

        stepio = popart.PyStepIO({input_tensor: input}, anchors)

        session.run(stepio)
        assert np.allclose(input, anchors["CustomopAttrPylist:0"])


def test_customOp_python_string():
    builder = popart.Builder()

    input_tensor = builder.addInputTensor(popart.TensorInfo("FLOAT", [1]))

    output_tensor = builder.customOp(
        opName="CustomopAttrPylist",
        opVersion=1,
        inputs=[input_tensor],
        domain="com.acme",
        attributes={"values": "Foo_Bar"},
    )[0]

    builder.addOutputTensor(output_tensor)
    proto = builder.getModelProto()

    # Create a runtime environment
    anchors = {output_tensor: popart.AnchorReturnType("All")}
    dataFlow = popart.DataFlow(1, anchors)

    with tu.create_test_device() as device:
        session = popart.InferenceSession(proto, dataFlow, device)

        session.prepareDevice()

        anchors = session.initAnchorArrays()

        input = np.random.rand(1).astype(np.float32)

        stepio = popart.PyStepIO({input_tensor: input}, anchors)

        session.run(stepio)
        assert np.allclose(input, anchors["CustomopAttrPylist:0"])


def test_customOp_python_int():
    builder = popart.Builder()

    input_tensor = builder.addInputTensor(popart.TensorInfo("FLOAT", [1]))

    output_tensor = builder.customOp(
        opName="CustomopAttrPylist",
        opVersion=1,
        inputs=[input_tensor],
        domain="com.acme",
        attributes={"values": 10},
    )[0]

    builder.addOutputTensor(output_tensor)
    proto = builder.getModelProto()

    # Create a runtime environment
    anchors = {output_tensor: popart.AnchorReturnType("All")}
    dataFlow = popart.DataFlow(1, anchors)

    with tu.create_test_device() as device:
        session = popart.InferenceSession(proto, dataFlow, device)

        session.prepareDevice()

        anchors = session.initAnchorArrays()

        input = np.random.rand(1).astype(np.float32)

        stepio = popart.PyStepIO({input_tensor: input}, anchors)

        session.run(stepio)
        assert np.allclose(input, anchors["CustomopAttrPylist:0"])


def test_customOp_python_float():
    builder = popart.Builder()

    input_tensor = builder.addInputTensor(popart.TensorInfo("FLOAT", [1]))

    output_tensor = builder.customOp(
        opName="CustomopAttrPylist",
        opVersion=1,
        inputs=[input_tensor],
        domain="com.acme",
        attributes={"values": 10.0},
    )[0]

    builder.addOutputTensor(output_tensor)
    proto = builder.getModelProto()

    # Create a runtime environment
    anchors = {output_tensor: popart.AnchorReturnType("All")}
    dataFlow = popart.DataFlow(1, anchors)

    with tu.create_test_device() as device:
        session = popart.InferenceSession(proto, dataFlow, device)

        session.prepareDevice()

        anchors = session.initAnchorArrays()

        input = np.random.rand(1).astype(np.float32)

        stepio = popart.PyStepIO({input_tensor: input}, anchors)

        session.run(stepio)
        assert np.allclose(input, anchors["CustomopAttrPylist:0"])
