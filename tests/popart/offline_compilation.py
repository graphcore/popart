# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import numpy as np
import os
import popart
import tempfile


def get_adding_model_session():
    d1 = np.random.rand(2, 2).astype(np.float32)
    d2 = np.random.rand(2, 2).astype(np.float32)
    d3 = np.random.rand(2, 2).astype(np.float32)
    d4 = np.random.rand(2, 2).astype(np.float32)

    datas = [np.random.rand(2, 2).astype(np.float32) for _ in range(4)]

    builder = popart.Builder()

    consts = [builder.aiOnnx.constant(data) for data in datas]
    a1 = builder.aiOnnx.add(consts[0:2])
    a2 = builder.aiOnnx.add(consts[2:4])
    a3 = builder.aiOnnx.add([a1, a2])

    out = a3
    builder.addOutputTensor(out)

    opts = {"numIPUs": 1}
    device = popart.DeviceManager().createOfflineIPUDevice(opts)

    session = popart.InferenceSession(
        fnModel=builder.getModelProto(),
        dataFlow=popart.DataFlow(1, {out: popart.AnchorReturnType("All")}),
        patterns=popart.Patterns(popart.PatternsLevel.All),
        deviceInfo=device)

    return session


def test_adding():
    session = get_adding_model_session()

    with tempfile.TemporaryDirectory() as tmpdirname:
        try:
            session.compileAndExport(tmpdirname, tmpdirname)
        except popart.popart_internal_exception as e:
            assert (str(e).endswith("Not yet implemented"))
        assert os.path.exists(tmpdirname)
        # The file for testing writing should be deleted
        assert (not os.path.exists(os.path.join(tmpdirname, "test_file")))
        #TODO, call the poplar runner

    # Try calling prepare device after
    try:
        session.prepareDevice()
    except popart.popart_exception as e:
        assert (str(e).endswith("Cannot run on an offline-ipu, " +
                                "use \"compileAndExport\" instead"))


def test_adding_none_paths():
    session = get_adding_model_session()

    try:
        session.compileAndExport(None, None)
    except popart.popart_internal_exception as e:
        assert (str(e).endswith("Not yet implemented"))


def test_adding_empty_str_paths():
    session = get_adding_model_session()

    try:
        session.compileAndExport("", "")
    except popart.popart_internal_exception as e:
        assert (str(e).endswith("Not yet implemented"))
    # #TODO, call the poplar runner
    # raise NotYetImplemented
