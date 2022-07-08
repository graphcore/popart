# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

# These are only failure case tests. See operators_test.py for happy path

import pytest
import popart
import numpy as np
import sys
from pathlib import Path

# `import test_util` requires adding to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu


def test_no_min_max_values_at_model_build():
    d1 = np.random.rand(2, 7).astype(np.float32)
    d1 = d1 * 6 - 3  # numbers in range [-3, 3]
    d_min = np.array([-1.5], dtype=np.float32)
    d_max = np.array([1.5], dtype=np.float32)

    builder = popart.Builder()
    i1 = builder.addInputTensor(popart.TensorInfo(d1), "d1")
    t_min = builder.addInputTensor(popart.TensorInfo(d_min), "tmin")
    t_max = builder.addInputTensor(popart.TensorInfo(d_max), "tmax")

    o = builder.aiOnnxOpset11.clip([i1, t_min, t_max], "MyClip11")

    builder.addOutputTensor(o)
    proto = builder.getModelProto()

    with tu.create_test_device() as device:
        anchor_desc = {o: popart.AnchorReturnType("ALL")}
        dataFlow = popart.DataFlow(5, anchor_desc)
        userOpts = popart.SessionOptions()

        with pytest.raises(Exception) as exceptionInfo:
            session = popart.InferenceSession(
                fnModel=proto,
                dataFlow=dataFlow,
                deviceInfo=device,
                userOptions=userOpts,
            )

            # Test currently raises exception on the above line, but below lines
            # may be useful when actually implementing dynamic thresholds

            session.prepareDevice()
            session.setRandomSeed(1)

            # Create buffers to receive results from the execution
            anchors = session.initAnchorArrays()
            boxes_npy = np.random.rand(1, 10, 7).astype(np.float32)
            min_npy = np.array([0.5], dtype=np.float32)
            max_npy = np.array([1.0], dtype=np.float32)

            inputs_dict = {"d1": boxes_npy, "tmin": min_npy, "tmax": max_npy}
            stepio = popart.PyStepIO(inputs_dict, anchors)
            session.run(stepio)

    assert (
        str(exceptionInfo.value)
        == "Op MyClip11(ai.onnx.Clip:11), inputs=[d1] currently only supports constant min/max parameters. Input 'min' (tmin) has no data."
    )


def test_no_min_value_at_model_build():
    d1 = np.random.rand(2, 7).astype(np.float32)
    d1 = d1 * 6 - 3  # numbers in range [-3, 3]
    d_min = np.array([-1.5], dtype=np.float32)
    d_max = np.array([1.5], dtype=np.float32)

    builder = popart.Builder()
    i1 = builder.addInputTensor(popart.TensorInfo(d1), "d1")
    t_min = builder.addInputTensor(popart.TensorInfo(d_min), "tmin")
    t_max = builder.aiOnnxOpset11.constant(d_max, False)

    o = builder.aiOnnxOpset11.clip([i1, t_min, t_max])

    builder.addOutputTensor(o)
    proto = builder.getModelProto()

    with tu.create_test_device() as device:
        anchor_desc = {o: popart.AnchorReturnType("ALL")}
        dataFlow = popart.DataFlow(5, anchor_desc)
        userOpts = popart.SessionOptions()

        with pytest.raises(Exception) as exceptionInfo:
            _ = popart.InferenceSession(
                fnModel=proto,
                dataFlow=dataFlow,
                deviceInfo=device,
                userOptions=userOpts,
            )

        assert (
            str(exceptionInfo.value)
            == "Op (ai.onnx.Clip:11), inputs=[d1] currently only supports constant min/max parameters. Input 'min' (tmin) has no data."
        )


def test_no_max_value_at_model_build():
    d1 = np.random.rand(2, 7).astype(np.float32)
    d1 = d1 * 6 - 3  # numbers in range [-3, 3]
    d_min = np.array([-1.5], dtype=np.float32)
    d_max = np.array([1.5], dtype=np.float32)

    builder = popart.Builder()
    i1 = builder.addInputTensor(popart.TensorInfo(d1), "d1")
    t_min = builder.aiOnnxOpset11.constant(d_min, False)
    t_max = builder.addInputTensor(popart.TensorInfo(d_max), "tmax")

    o = builder.aiOnnxOpset11.clip([i1, t_min, t_max])

    builder.addOutputTensor(o)
    proto = builder.getModelProto()

    with tu.create_test_device() as device:
        anchor_desc = {o: popart.AnchorReturnType("ALL")}
        dataFlow = popart.DataFlow(5, anchor_desc)
        userOpts = popart.SessionOptions()

        with pytest.raises(Exception) as exceptionInfo:
            _ = popart.InferenceSession(
                fnModel=proto,
                dataFlow=dataFlow,
                deviceInfo=device,
                userOptions=userOpts,
            )

        assert (
            str(exceptionInfo.value)
            == "Op (ai.onnx.Clip:11), inputs=[d1] currently only supports constant min/max parameters. Input 'max' (tmax) has no data."
        )
