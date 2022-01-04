# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart
import onnx
from onnx import numpy_helper
import numpy as np
import pytest


def test_np_memory_layout_add_initialized_input_tensor1():
    """ Test that when we create a parameter input with a non-contiguous array
        things still work (first test).
    """
    np.random.seed(1)

    # Build a computational graph. Initialise an input parameter with a transposed
    # input (which happens to be non-contiguous in numpy).
    builder = popart.Builder()
    input1Value = np.random.randint(0, 100, size=(2, 3), dtype='int32')
    input1Value = np.transpose(input1Value, [1, 0])
    input1 = builder.addInitializedInputTensor(input1Value)
    input1 = builder.aiOnnx.identity([input1])
    builder.addOutputTensor(input1)

    # Perpare a session.
    anchorConfig = {input1: popart.AnchorReturnType("ALL")}
    dataFlow = popart.DataFlow(1, anchorConfig)
    deviceConfig = {'numIPUs': 1}
    dm = popart.DeviceManager()
    device = dm.createIpuModelDevice(deviceConfig)
    session = popart.InferenceSession(fnModel=builder.getModelProto(),
                                      dataFlow=dataFlow,
                                      deviceInfo=device)

    session.prepareDevice()
    session.weightsFromHost()
    anchors = session.initAnchorArrays()

    # Run the session.
    stepio = popart.PyStepIO({}, anchors)
    session.run(stepio)

    # Compare outputs.
    assert (anchors[input1] == input1Value
            ).all(), f"Expected {anchors[input1]} to match {input1Value}"


def test_np_memory_layout_add_initialized_input_tensor2():
    """ Test that when we create a parameter input with a non-contiguous array
        it is correctly represented in the computational graph.
    """
    np.random.seed(1)

    # Build a computational graph. Add to input parameters, one contiguous and one transposed
    # and hence non-contiguous.
    builder = popart.Builder()
    input1Value = np.random.randint(0, 100, size=(3, 5), dtype='int32')
    input2Value = np.random.randint(0, 100, size=(20, 30), dtype='int32')
    input2Value = np.transpose(input2Value)
    _ = builder.addInitializedInputTensor(input1Value, "contiguous_input")
    _ = builder.addInitializedInputTensor(input2Value, "transposed_input")

    # Get data back from computational graph.
    proto = builder.getModelProto()
    onnxProto = onnx.load_model_from_string(proto)
    input1Init = next(arr for arr in onnxProto.graph.initializer
                      if 'contiguous_input' in arr.name)
    input2Init = next(arr for arr in onnxProto.graph.initializer
                      if 'transposed_input' in arr.name)
    input1ValueInGraph = numpy_helper.to_array(input1Init)
    input2ValueInGraph = numpy_helper.to_array(input2Init)

    # Test the data matches the initialised arrays.
    assert (input1ValueInGraph == input1Value
            ).all(), f"Expected {input1ValueInGraph} to match {input1Value}"
    assert (input2ValueInGraph == input2Value
            ).all(), f"Expected {input2ValueInGraph} to match {input2Value}"


def test_np_memory_layout_add_constant():
    """ Test that when we create a constant tensor with a non-contiguous array
        things still work.
    """
    np.random.seed(1)

    # Create a computational graph in which a constant input is given a
    # non-contiguous array to as a value.
    builder = popart.Builder()
    constant1Value = np.random.randint(0, 100, size=(2, 2), dtype='int32')
    constant1Value = np.transpose(constant1Value, [1, 0])
    constant1 = builder.aiOnnx.constant(constant1Value)

    # Run a session to prove this
    output1 = builder.aiOnnx.identity([constant1])
    builder.addOutputTensor(output1)
    anchorConfig = {output1: popart.AnchorReturnType("ALL")}

    dataFlow = popart.DataFlow(1, anchorConfig)
    deviceConfig = {'numIPUs': 1}
    dm = popart.DeviceManager()
    device = dm.createIpuModelDevice(deviceConfig)
    session = popart.InferenceSession(fnModel=builder.getModelProto(),
                                      dataFlow=dataFlow,
                                      deviceInfo=device)

    # Compile graph and place weights onto it
    session.prepareDevice()
    session.weightsFromHost()
    anchors = session.initAnchorArrays()

    # Feed
    stepio = popart.PyStepIO({}, anchors)
    session.run(stepio)

    # This assertion fails
    assert (anchors[output1] == constant1Value
            ).all(), f"Expected {anchors[output1]} to match {constant1Value}"


def test_np_memory_layout_add_input_tensor_pystepio():
    """ In the case of input / output data a conversion could have significant
        impact on performance and hence we do not allow it. Here, we test it is
        detected and an error is thrown. 

        NOTE: We don't test non-contiguous outputs in this test as they
        are usually made by a call to initAnchorArrays anyway.
    """

    builder = popart.Builder()

    # Create a random constant and transpose it
    np.random.seed(1)
    input1 = builder.addInputTensor("INT32", [2, 2])

    # Run a session to prove this
    output1 = builder.aiOnnx.identity([input1])
    builder.addOutputTensor(output1)
    anchorConfig = {output1: popart.AnchorReturnType("ALL")}

    dataFlow = popart.DataFlow(1, anchorConfig)
    deviceConfig = {'numIPUs': 1}
    dm = popart.DeviceManager()
    device = dm.createIpuModelDevice(deviceConfig)
    session = popart.InferenceSession(fnModel=builder.getModelProto(),
                                      dataFlow=dataFlow,
                                      deviceInfo=device)

    # Compile graph and place weights onto it
    session.prepareDevice()
    session.weightsFromHost()
    anchors = session.initAnchorArrays()

    # Feed the session with a transposed (non-contiguous) tensor.
    input1Value = np.random.randint(0, 100, size=(2, 2), dtype='int32')
    input1Value = np.transpose(input1Value, [1, 0])

    with pytest.raises((RuntimeError, popart.popart_exception)) as e_info:
        stepio = popart.PyStepIO({input1: input1Value}, anchors)
        session.run(stepio)

    assert "contiguous" in e_info.value.args[0]


def test_np_memory_layout_add_input_tensor_pystepiocallback():
    """ In the case of input / output data a conversion could have significant
        impact on performance and hence we do not allow it. Here, we test it is
        detected and an error is thrown. 
    """

    def _test(transposedInput, transposedOutput):
        builder = popart.Builder()

        # Create a random constant and transpose it
        np.random.seed(1)
        input1 = builder.addInputTensor("INT32", [2, 2])

        # Run a session to prove this
        output1 = builder.aiOnnx.identity([input1])
        builder.addOutputTensor(output1)
        anchorConfig = {output1: popart.AnchorReturnType("ALL")}

        dataFlow = popart.DataFlow(1, anchorConfig)
        deviceConfig = {'numIPUs': 1}
        dm = popart.DeviceManager()
        device = dm.createIpuModelDevice(deviceConfig)
        session = popart.InferenceSession(fnModel=builder.getModelProto(),
                                          dataFlow=dataFlow,
                                          deviceInfo=device)

        # Compile graph and place weights onto it
        session.prepareDevice()
        session.weightsFromHost()

        # Feed the session with a transposed (non-contiguous) tensor.
        input1Value = np.random.randint(0, 100, size=(2, 2), dtype='int32')
        if transposedInput:
            input1Value = np.transpose(input1Value, [1, 0])
        output1Value = np.random.randint(0, 100, size=(2, 2), dtype='int32')
        if transposedOutput:
            output1Value = np.transpose(output1Value, [1, 0])

        with pytest.raises(
            (Exception, RuntimeError, popart.popart_exception)) as e_info:

            def input_callback(id, prefetch):
                return input1Value

            def input_complete_callback(id):
                pass

            def output_callback(id):
                return output1Value

            def output_complete_callback(id):
                pass

            stepio = popart.PyStepIOCallback(
                input_callback, input_complete_callback, output_callback,
                output_complete_callback)

            session.run(stepio)

        assert "contiguous" in e_info.value.args[0]

    _test(transposedInput=True, transposedOutput=False)
    _test(transposedInput=False, transposedOutput=True)
