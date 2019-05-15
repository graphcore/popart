import poponnx
import test_util as tu


# test that the input tensors of add have been mapped to different tiles
def test_basic_mapping(tmpdir):

    builder = poponnx.Builder()

    shape = poponnx.TensorInfo("FLOAT", [512])

    i1 = builder.addInputTensor(shape)
    i2 = builder.addInputTensor(shape)
    o = builder.aiOnnx.add([i1, i2])
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    dataFlow = poponnx.DataFlow(1, {o: poponnx.AnchorReturnType("ALL")})

    session = poponnx.InferenceSession(
        fnModel=proto,
        dataFeed=dataFlow,
        deviceInfo=tu.get_ipu_model(compileIPUCode=False))

    anchors = session.initAnchorArrays()

    session.prepareDevice()

    m = session.getTensorTileMap()

    # get tile mappings of i1 and i2 as a Dict[Tile, Intervals]
    i1_map = {t: i for t, i in enumerate(m[i1]) if i}
    i2_map = {t: i for t, i in enumerate(m[i2]) if i}

    # i1 and i2 maps should not share any tiles
    assert set(i1_map.keys()).isdisjoint(set(i2_map.keys()))
