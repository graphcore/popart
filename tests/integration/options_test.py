# Copyright (c) 2018 Graphcore Ltd. All rights reserved.
import popart
import pytest
import test_util as tu


def test_create_empty_options():

    opts = popart.SessionOptions()
    assert (opts is not None)
    # assert that the dotChecks set is empty:
    assert (not opts.dotChecks)
    assert (len(opts.engineOptions) == 0)
    assert (len(opts.convolutionOptions) == 0)
    assert (len(opts.reportOptions) == 0)
    assert (opts.logDir == "")


def test_set_dotchecks():

    opts = popart.SessionOptions()
    assert (len(opts.dotChecks) == 0)

    opts.dotChecks = {"Final", "Fwd0"}
    assert (len(opts.dotChecks) == 2)


def test_set_enabledOutlining_flag():

    opts = popart.SessionOptions()
    assert (opts.enableOutlining == True)

    opts.enableOutlining = False
    assert (opts.enableOutlining == False)


def test_set_engineOptions():

    opts = popart.SessionOptions()
    opts.engineOptions = {'option': 'value'}

    assert (len(opts.engineOptions) == 1)
    assert (len(opts.convolutionOptions) == 0)
    assert (len(opts.reportOptions) == 0)
    assert (opts.engineOptions['option'] == 'value')


def test_set_convolutionOptions():

    opts = popart.SessionOptions()
    opts.convolutionOptions = {'option': 'value'}

    assert (len(opts.engineOptions) == 0)
    assert (len(opts.convolutionOptions) == 1)
    assert (len(opts.reportOptions) == 0)
    assert (opts.convolutionOptions['option'] == 'value')


def test_set_reportOptions():

    opts = popart.SessionOptions()
    opts.reportOptions = {'option': 'value'}

    assert (len(opts.engineOptions) == 0)
    assert (len(opts.convolutionOptions) == 0)
    assert (len(opts.reportOptions) == 1)
    assert (opts.reportOptions['option'] == 'value')


def test_engine_options_passed_to_engine():

    popart.getLogger().setLevel("DEBUG")

    builder = popart.Builder()

    shape = popart.TensorInfo("FLOAT", [1, 2, 32, 32])

    i1 = builder.addInputTensor(shape)
    i2 = builder.addInputTensor(shape)

    o = builder.aiOnnx.add([i1, i2])

    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    dataFlow = popart.DataFlow(1, {o: popart.AnchorReturnType("All")})

    opts = popart.SessionOptions()
    opts.engineOptions = {'option': 'value'}

    with tu.create_test_device() as device:
        with pytest.raises(popart.poplar_exception) as e_info:
            session = popart.InferenceSession(fnModel=proto,
                                              dataFlow=dataFlow,
                                              userOptions=opts,
                                              deviceInfo=device)

            session.initAnchorArrays()
            session.prepareDevice()

    assert (e_info.value.args[0].endswith("Unrecognised option 'option'"))


# An error should be thrown if the user tries to set an unrecognised option flag.
# This is to prevent confusion is a user has a typo in an flag.
def test_set_bad_options():
    opts = popart.SessionOptions()
    with pytest.raises(AttributeError):
        opts.foo = 'bar'


def test_update_options():
    opts = popart.SessionOptions()
    opts.engineOptions = {"aaaaa": "a"}
    opts.engineOptions["bbbb"] = "b"

    pdict = {"aaaaa": "a"}
    pdict["bbbb"] = "b"

    print("\nEngine options dict:")
    for opt, val in opts.engineOptions.items():
        print("opt:", opt, "val:", val)

    print("\nPython dict:")
    for opt, val in pdict.items():
        print("opt:", opt, "val:", val)

    # assert that the dictionaries are the same
    keys_a = set(opts.engineOptions.keys())
    keys_b = set(pdict.keys())
    assert keys_a == keys_b

    for key in keys_a:
        a = opts.engineOptions[key]
        b = pdict[key]
        assert a == b


# Check we can read from and write to numIOTiles from python
def test_numIOTiles_option():
    opts = popart.SessionOptions()
    opts.numIOTiles = 42
    assert opts.numIOTiles == 42
