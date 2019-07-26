import os
import popart
import test_util as tu
import json


# test that the input tensors of add have been mapped to different tiles
def test_basic_mapping(tmpdir):

    builder = popart.Builder()

    shape = popart.TensorInfo("FLOAT", [512])

    i1 = builder.addInputTensor(shape)
    i2 = builder.addInputTensor(shape)
    o = builder.aiOnnx.add([i1, i2])
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    dataFlow = popart.DataFlow(1, {o: popart.AnchorReturnType("ALL")})

    session = popart.InferenceSession(
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


# test that the tile mapping can be saved using the environment variable
def test_environment_mapping(tmpdir):
    mapFileName = 'ttm.js'
    os.environ['POPART_TENSOR_TILE_MAP'] = mapFileName

    builder = popart.Builder()

    shape = popart.TensorInfo("FLOAT", [512])

    i1 = builder.addInputTensor(shape)
    i2 = builder.addInputTensor(shape)
    o = builder.aiOnnx.add([i1, i2])
    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    dataFlow = popart.DataFlow(1, {o: popart.AnchorReturnType("ALL")})

    opts = popart.SessionOptions()
    opts.logDir = str(tmpdir)

    session = popart.InferenceSession(
        fnModel=proto,
        dataFeed=dataFlow,
        userOptions=opts,
        deviceInfo=tu.get_ipu_model(compileIPUCode=False))

    anchors = session.initAnchorArrays()

    session.prepareDevice()

    with open(tmpdir / mapFileName, 'r') as f:
        m = json.load(f)

    assert set(m.keys()) == set([i1, i2, o])

    # get tile mappings of i1 and i2 as a Dict[Tile, Intervals]
    i1_map = {t: i for t, i in enumerate(m[i1]) if i}
    i2_map = {t: i for t, i in enumerate(m[i2]) if i}

    # i1 and i2 maps should not share any tiles
    assert set(i1_map.keys()).isdisjoint(set(i2_map.keys()))
