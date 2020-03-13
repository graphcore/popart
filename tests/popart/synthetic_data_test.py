# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
import pytest

# `import test_util` requires adding to sys.path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu

d_shape = [100]
np.random.seed(0)


def run_pt_session(syntheticDataMode):
    builder = popart.Builder()
    d0 = builder.addInputTensor(popart.TensorInfo("FLOAT", d_shape))
    p = builder.aiGraphcore.printtensor([d0])

    opts = popart.SessionOptions()
    opts.syntheticDataMode = syntheticDataMode

    session = popart.InferenceSession(fnModel=builder.getModelProto(),
                                      dataFeed=popart.DataFlow(
                                          1,
                                          {p: popart.AnchorReturnType("ALL")}),
                                      userOptions=opts,
                                      deviceInfo=tu.create_test_device())

    session.prepareDevice()
    anchors = session.initAnchorArrays()
    stepio = popart.PyStepIO({d0: np.ones(d_shape)}, anchors)
    session.run(stepio)


def numpy_array_from_printtensor_string(string):
    stringData = string.partition('{')[2].partition('}')[0]
    data = np.fromstring(stringData, dtype=float, sep=',')
    print(data)
    return data


# TODO see T16010
# @tu.requires_ipu
@pytest.mark.skip("Test currently failing on hardware")
def test_verify_synthetic_inputs(capfd):
    """
    For each synthetic data mode:
    1. Get a session that prints the input tensor value to stderr
    2. Capture the tensor data from stderr
    3. Verify that the data is as expected for that synthetic data mode
    """

    # Test depends on logging output. Silence the logging from PopART
    popart.getLogger().setLevel("OFF")

    ## A) Expect input is all zeros
    run_pt_session(popart.SyntheticDataMode.Zeros)
    _, err0 = capfd.readouterr()
    zeroData = numpy_array_from_printtensor_string(err0)
    assert np.all(zeroData == 0)

    ## B) Expect input is random normal, T~N(0,1)
    run_pt_session(popart.SyntheticDataMode.RandomNormal)
    _, err1 = capfd.readouterr()
    rnData = numpy_array_from_printtensor_string(err1)
    assert np.all(rnData == 0) == False
    assert np.isclose(np.mean(rnData), 0, atol=0.02)
    assert np.isclose(np.std(rnData), 1, atol=0.1)
