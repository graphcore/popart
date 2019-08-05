import pytest
import popart
import test_util as tu


def test_set_random_seed_error():

    popart.getLogger().setLevel("TRACE")

    builder = popart.Builder()

    i1 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1]))
    o = builder.aiOnnx.add([i1, i1])
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    dataFlow = popart.DataFlow(1, {o: popart.AnchorReturnType("ALL")})

    s = popart.InferenceSession(
        fnModel=proto,
        dataFeed=dataFlow,
        userOptions=popart.SessionOptionsCore(),
        deviceInfo=tu.get_ipu_model(numIPUs=2))

    with pytest.raises(popart.popart_exception) as e_info:
        s.setRandomSeed(0)

    msg = e_info.value.args[0]
    assert msg == ("Devicex::prepare() must be called before "
                   "Devicex::setRandomSeed(uint64_t) is called.")
