# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
import pytest

# `import test_util` requires adding to sys.path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu


@tu.requires_ipu
def test_rng_set_and_get():
    """
    1. Create a training and validation session 
        with the option to enable rng set/get.
    2. Get the initial RNG state values
    3. Step 1 : do 5 runs of the training session, twice
    4. Step 2 : 
        - reset the RNG to the initial state
        - do 5 runs of the training session 
        - capture the rng state
        - do 1 run of the validation session
        - restore the rng
        - do 5 runs of the training session again
    5. Step 3 :
        - Reset the RNG to the initial state
        - do 5 runs of the training session, 
        - do 1 run of the validation session
        - do 5 runs of the training session again
    6. Results comparison:
        Steps 1 and 2 must have the same outputs
        after the series of 5 runs.
        Step 3 must have a different output after the second 
        series of 5 runs, due to session overwritting RNG state.
    """

    np.random.seed(0)

    # Model definition
    builder = popart.Builder()
    dShape = [100, 100]
    i0 = builder.addInputTensor(popart.TensorInfo("FLOAT16", dShape))
    wData = np.random.rand(*dShape).astype(np.float16)
    w0 = builder.addInitializedInputTensor(wData)
    out = builder.aiOnnx.matmul([i0, w0])
    loss = builder.aiGraphcore.l1loss([out], 0.1)

    device = tu.create_test_device(1)

    # Enable the options
    options = popart.SessionOptions()
    options.enableLoadAndOffloadRNGState = True
    options.enableStochasticRounding = True
    options.constantWeights = False
    options._enableRngStateManagement = True

    # Training session
    bps = 5
    tr_opt = popart.SGD({"defaultMomentum": (0.01, True)})
    session = popart.TrainingSession(fnModel=builder.getModelProto(),
                                     dataFlow=popart.DataFlow(bps, [out]),
                                     loss=loss,
                                     optimizer=tr_opt,
                                     deviceInfo=device,
                                     userOptions=options)
    session.prepareDevice()
    anchors = session.initAnchorArrays()

    # Get the initial RNG state before any other operation.
    init_rng = session.getRNGState()

    # Interfering inference session
    interfering_session = popart.InferenceSession(
        fnModel=builder.getModelProto(),
        dataFlow=popart.DataFlow(bps, [out]),
        deviceInfo=device,
        userOptions=options)
    interfering_session.prepareDevice()
    inf_anchors = interfering_session.initAnchorArrays()

    # Input data
    data_a = np.random.rand(5, 100, 100).astype(np.float16)

    def run_session(session):
        stepio = popart.PyStepIO({i0: data_a}, anchors)
        session.run(stepio)
        return session.getRNGState(), anchors['MatMul:0'].tolist()

    def run_interference(interfering_session):
        interfering_session.weightsFromHost()
        inf_stepio = popart.PyStepIO({i0: data_a}, inf_anchors)
        interfering_session.run(inf_stepio)

    # Step 1 -> training, training
    session.weightsFromHost()
    session.setRNGState(init_rng)
    rng, pre1 = run_session(session)
    session.weightsFromHost()
    rng2, output1 = run_session(session)
    assert rng != rng2

    # Step 2 -> interleaved training, validation, training
    session.weightsFromHost()
    session.setRNGState(init_rng)
    rng, pre2 = run_session(session)
    run_interference(interfering_session)
    session.weightsFromHost()
    session.setRNGState(rng)
    rng2, output2 = run_session(session)
    assert output1 == output2

    # Step 3 -> interleaved training, validation, RNG not restored
    session.weightsFromHost()
    session.setRNGState(init_rng)
    rng, pre3 = run_session(session)
    run_interference(interfering_session)
    session.weightsFromHost()
    rng2, output3 = run_session(session)

    assert pre1 == pre2 == pre3
    assert (output3 != output1)

    # Small tests about the seed
    init_rng = session.getRNGState()

    # not all states are valid, but we don't check that as long as the size is correct
    new_rng = [k for k in range(len(init_rng))]
    session.setRNGState(new_rng)
    rng1 = session.getRNGState()
    assert (rng1 == new_rng)

    session.setRNGState(init_rng)
    rng2 = session.getRNGState()
    assert (rng2 == init_rng)

    # check that an RNGState of the wrong size raises an exception
    init_rng.append(0)
    with pytest.raises(popart.popart_exception) as e_info:
        session.setRNGState(init_rng)
    assert e_info.value.args[0].startswith(
        "Devicex::setRngStateValue received rngState of size")
