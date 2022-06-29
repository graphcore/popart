# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
from collections import namedtuple
import numpy as np
import popart
import pytest

# `import test_util` requires adding to sys.path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu

np.random.seed(0)

_DataType = namedtuple('_DataType', ['builder_type', 'np_type'])
_INT8 = _DataType('INT8', np.int8)
_UINT8 = _DataType('UINT8', np.uint8)


def run_pt_session(syntheticDataMode, inputType=None, d_shape=[100]):
    builder = popart.Builder()
    if inputType is not None:
        d0_i8 = builder.addInputTensor(
            popart.TensorInfo(inputType.builder_type, d_shape))
        d0 = builder.aiOnnx.cast([d0_i8], "FLOAT")
        in_name = d0_i8
    else:
        d0 = builder.addInputTensor(popart.TensorInfo("FLOAT", d_shape))
        in_name = d0
    p = builder.aiGraphcore.printtensor([d0])

    opts = popart.SessionOptions()
    opts.syntheticDataMode = syntheticDataMode

    with tu.create_test_device() as device:
        session = popart.InferenceSession(fnModel=builder.getModelProto(),
                                          dataFlow=popart.DataFlow(1, [p]),
                                          userOptions=opts,
                                          deviceInfo=device)

        session.prepareDevice()
        anchors = session.initAnchorArrays()
        stepio = popart.PyStepIO({in_name: np.ones(d_shape)}, anchors)
        session.run(stepio)


def numpy_array_from_printtensor_string(string):
    stringData = string.partition('{')[2].partition('}')[0]
    data = np.fromstring(stringData, dtype=float, sep=',')
    print(data)
    return data


@tu.requires_ipu
@pytest.mark.parametrize("inputType", [_INT8, _UINT8, None])
def test_verify_synthetic_inputs(capfd, inputType):
    """
    For each synthetic data mode:
    1. Get a session that prints the input tensor value to stderr
    2. Capture the tensor data from stderr
    3. Verify that the data is as expected for that synthetic data mode
    """

    # Hopefully this is large enough to achieve desired tolerance for mean/std,
    # even for ints.
    d_shape = [4000]

    # Test depends on logging output. Silence the logging from PopART
    popart.getLogger().setLevel("OFF")

    ## A) Expect input is all zeros
    run_pt_session(popart.SyntheticDataMode.Zeros,
                   inputType=inputType,
                   d_shape=d_shape)
    _, err0 = capfd.readouterr()
    zeroData = numpy_array_from_printtensor_string(err0)
    assert np.all(zeroData == 0)

    ## B) Expect input is random normal, T~N(0,1)
    if inputType == _UINT8:
        # Casting normal data to unsigned results in non-normal data.
        return

    run_pt_session(popart.SyntheticDataMode.RandomNormal,
                   inputType=inputType,
                   d_shape=d_shape)
    _, err1 = capfd.readouterr()
    rnData = numpy_array_from_printtensor_string(err1)

    assert np.all(rnData == 0) == False
    assert np.isclose(np.mean(rnData), 0, atol=0.02)
    assert np.isclose(np.std(rnData), 1, atol=0.1)


def test_supported_input_type_float16():
    def run_with_input_of_type(dtype):
        builder = popart.Builder()
        in0 = builder.addInputTensor(popart.TensorInfo(dtype, [2]))
        out = builder.aiOnnx.sqrt([in0])

        opts = popart.SessionOptions()
        opts.syntheticDataMode = popart.SyntheticDataMode.RandomNormal
        _ = popart.InferenceSession(
            fnModel=builder.getModelProto(),
            userOptions=opts,
            deviceInfo=popart.DeviceManager().createCpuDevice(),
            dataFlow=popart.DataFlow(1, [out]))

    run_with_input_of_type("FLOAT16")
    run_with_input_of_type("FLOAT")
    run_with_input_of_type("INT32")
    run_with_input_of_type("UINT32")
