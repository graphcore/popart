# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
import pytest

# `import test_util` requires adding to sys.path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu


def get_simple_model_cycle_count(bps):
    builder = popart.Builder()
    # Make the model large enough such that the cycle count is dominated
    # by compute and internal exchange (as apposed to host exchage)
    d_shape = [200, 200]
    d0 = builder.addInputTensor(popart.TensorInfo("FLOAT", d_shape))
    out = d0
    for _ in range(100):
        out = builder.aiOnnx.sin([out])

    opts = popart.SessionOptions()
    opts.instrumentWithHardwareCycleCounter = True
    # Verify that we can still measure cycles when data streams
    # (inuts/weights/anchors) are off
    opts.syntheticDataMode = popart.SyntheticDataMode.Zeros
    patterns = popart.Patterns(popart.PatternsLevel.NoPatterns)
    patterns.enableRuntimeAsserts(False)

    with tu.create_test_device() as device:
        session = popart.InferenceSession(
            fnModel=builder.getModelProto(),
            dataFlow=popart.DataFlow(bps,
                                     {out: popart.AnchorReturnType("All")}),
            userOptions=opts,
            deviceInfo=device,
            patterns=patterns)

        session.prepareDevice()
        anchors = session.initAnchorArrays()
        if bps > 1:
            d_shape.insert(0, bps)
        stepio = popart.PyStepIO(
            {d0: np.random.rand(*d_shape).astype(np.float32)}, anchors)
        session.run(stepio)

        cycles = session.getCycleCount()
        cycles_ = session.getCycleCount()
    print("BPS: ", bps, " Cycles: ", cycles)
    # Verify that the tensor is not overwritten when streaming off device
    assert (cycles == cycles_)
    return cycles


@tu.requires_ipu
def test_check_sensible_cycle_counts():
    cycles100 = get_simple_model_cycle_count(bps=100)
    cycles100_ = get_simple_model_cycle_count(bps=100)
    cycles200 = get_simple_model_cycle_count(bps=200)
    cycles300 = get_simple_model_cycle_count(bps=300)
    diff_100_200 = cycles200 - cycles100
    diff_200_300 = cycles300 - cycles200

    # Expecting consistent results:
    # 1. Cycles are similar* between runs for the same bps
    assert abs(cycles100 - cycles100_) / cycles100_ < 0.15
    # 2. Cycles scales roughly* linearly with bps
    assert abs(diff_100_200 - diff_200_300) / diff_200_300 < 0.15

    # (*) Note: Since cycle count includes cycles when waiting for host syncs,
    #     cannot expect exact results due to variability in host communication


@tu.requires_ipu
def test_get_cycle_count_requires_run():
    builder = popart.Builder()
    d0 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1]))
    p = builder.aiOnnx.exp([d0])

    opts = popart.SessionOptions()
    opts.instrumentWithHardwareCycleCounter = True

    with tu.create_test_device() as d:
        session = popart.InferenceSession(fnModel=builder.getModelProto(),
                                          dataFlow=popart.DataFlow(1, [p]),
                                          userOptions=opts,
                                          deviceInfo=d)
        session.prepareDevice()

        with pytest.raises(popart.popart_exception) as e_info:
            _ = session.getCycleCount()
        assert e_info.value.args[0].startswith(
            "Must call run before getCycleCount")


@tu.requires_ipu
def test_get_cycle_count_requires_instrumentation_option():
    builder = popart.Builder()
    d0 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1]))
    p = builder.aiOnnx.exp([d0])

    # Default SessionOptions - cycle count instrumentation off
    with tu.create_test_device() as d:
        session = popart.InferenceSession(fnModel=builder.getModelProto(),
                                          dataFlow=popart.DataFlow(1, [p]),
                                          deviceInfo=d)
        session.prepareDevice()
        stepio = popart.PyStepIO({d0: np.random.rand(1).astype(np.float32)},
                                 session.initAnchorArrays())
        session.run(stepio)

        with pytest.raises(popart.popart_exception) as e_info:
            _ = session.getCycleCount()
        assert e_info.value.args[0].startswith(
            "SessionOption 'instrumentWithHardwareCycleCounter' must be")


@tu.requires_ipu
def test_get_cycle_count_bad_id():
    builder = popart.Builder()
    d0 = builder.addInputTensor("FLOAT", [1])
    p = builder.aiOnnx.exp([d0])

    def getInstrumentedSession(instrumentation, device):
        opts = popart.SessionOptions()
        opts.instrumentWithHardwareCycleCounter = True
        opts.hardwareInstrumentations = instrumentation
        session = popart.InferenceSession(fnModel=builder.getModelProto(),
                                          dataFlow=popart.DataFlow(1, [p]),
                                          userOptions=opts,
                                          deviceInfo=device)
        session.prepareDevice()
        stepio = popart.PyStepIO({d0: np.random.rand(1).astype(np.float32)},
                                 session.initAnchorArrays())
        session.run(stepio)
        return session

    with tu.create_test_device() as d:
        s = getInstrumentedSession({popart.Instrumentation.Outer}, d)
        with pytest.raises(popart.popart_exception) as e_info:
            _ = s.getCycleCount("inner_ipu_0")
        assert e_info.value.args[0].endswith(
            "Make sure you have set SessionOption::hardwareInstrumentations correctly."
        )

    with tu.create_test_device() as d:
        s = getInstrumentedSession({popart.Instrumentation.Inner}, d)
        _ = s.getCycleCount("inner_ipu_0")


@tu.requires_ipu
@pytest.mark.parametrize("useIOTiles", [True, False])
def test_get_cycle_count_replication(useIOTiles):
    builder = popart.Builder()
    d0 = builder.addInputTensor("FLOAT", [1])
    with builder.virtualGraph(0):
        act = builder.aiOnnx.exp([d0])
    with builder.virtualGraph(1):
        act = builder.aiOnnx.sin([act])

    def getInstrumentedSession(instrumentation, device):
        opts = popart.SessionOptions()
        opts.instrumentWithHardwareCycleCounter = True
        opts.hardwareInstrumentations = instrumentation
        opts.replicatedGraphCount = 2
        opts.virtualGraphMode = popart.VirtualGraphMode.Manual
        opts.enableReplicatedGraphs = True
        if useIOTiles is True:
            opts.numIOTiles = 32
        session = popart.InferenceSession(fnModel=builder.getModelProto(),
                                          dataFlow=popart.DataFlow(20, [act]),
                                          userOptions=opts,
                                          deviceInfo=device)
        session.prepareDevice()
        stepio = popart.PyStepIO({d0: np.random.rand(40).astype(np.float32)},
                                 session.initAnchorArrays())
        session.run(stepio)
        return session

    if useIOTiles is True:
        # Trying to use less than all the tiles throw an error like
        #   popart_core.poplar_exception: Trying to access tile 72 on IPU
        #   0 but the virtual graph only covers the following tiles on
        #   that IPU: 0-63
        # The error happens in a call to poplar made by gcl::perIPUTiles.
        tilesPerIPU = tu.USE_ALL_TILES
    else:
        tilesPerIPU = 4
    with tu.create_test_device(numIpus=4, tilesPerIPU=tilesPerIPU) as device:
        s = getInstrumentedSession(
            {popart.Instrumentation.Outer, popart.Instrumentation.Inner},
            device)
        print(s.getCycleCount())
        print(s.getCycleCount("inner_ipu_0"))
        print(s.getCycleCount("inner_ipu_1"))
