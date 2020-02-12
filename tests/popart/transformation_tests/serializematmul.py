import numpy as np
import pytest
import popart
import pprint
import json
import platform

# 'import test_util' requires adding to sys.path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu


def getBaseOptions():
    opts = popart.SessionOptions()
    opts.reportOptions = {"showExecutionSteps": "true"}
    opts.enableGroupedMatmuls = False
    # TODO(T14786) investigate why swapping causes some tests to fail
    opts.swapLimitScheduler = -1
    # TODO(T14786) investigate why GREEDY causes some tests to fail
    opts.kahnTieBreaker = "FIFO"
    return opts


def _is_grad_tensor(x):
    return x['name'].startswith(popart.reservedGradientPrefix())


# The backward lhs matmul should take a gradient tensor in
# as input 0 and a non gradient tensor in as input 1
def _get_bwd_lhs_matmul(matmuls):
    result = []
    for m in matmuls:
        i0 = m['inputs'][0]
        i1 = m['inputs'][1]
        if _is_grad_tensor(i0) and not _is_grad_tensor(i1):
            result.append(m)
    assert len(result) == 1
    return result[0]


# The backward rhs matmul should take a non gradient tensor
# in as input 0 and a gradient tensor in as input  1
def _get_bwd_rhs_matmul(matmuls):
    result = []
    for m in matmuls:
        i0 = m['inputs'][0]
        i1 = m['inputs'][1]
        if not _is_grad_tensor(i0) and _is_grad_tensor(i1):
            result.append(m)
    assert len(result) == 1
    return result[0]


def gen_shape(shape):
    return '[{0} {1} {2}]'.format(str(shape[0]), str(shape[1]), str(shape[2]))


def test_matmul_serialization_invalid_mode(tmpdir):

    if platform.system() == "Darwin":
        # MacOS is throwing a RuntimeError not popart_exception
        print("T11614 : skipping this test on mac/os")
        pytest.skip("T11614 : skipping this test on mac/os")
    else:
        lhs_shape = [2, 2]
        rhs_shape = [2, 4]
        lhs_data = np.random.rand(*lhs_shape).astype(np.float32)
        rhs_data = np.random.rand(*rhs_shape).astype(np.float32)

        builder = popart.Builder()

        lhs = builder.addInputTensor(popart.TensorInfo("FLOAT", lhs_shape),
                                     "lhs")
        rhs = builder.addInputTensor(popart.TensorInfo("FLOAT", rhs_shape),
                                     "rhs")

        o = builder.aiOnnx.matmul([lhs, rhs])

        with pytest.raises(popart.popart_exception) as e_info:
            try:
                builder.setSerializeMatMul({o}, "invalid_mode")
            except popart.popart_exception as e:
                print("Catch popart_exception ", type(e))
                raise
            except Exception as e:
                print("Unexpected exception from setSerializeMatMul ", type(e))
                raise

        assert (e_info.value.args[0].startswith(
            "Unsupported mat mul serialization mode 'invalid_mode'. Supported modes are 'input_channels', 'reducing_dim', 'output_channels' or 'none'"
        ))


def test_matmul_serialization_invalid_factor(tmpdir):
    lhs_shape = [2, 2]
    rhs_shape = [2, 4]
    lhs_data = np.random.rand(*lhs_shape).astype(np.float32)
    rhs_data = np.random.rand(*rhs_shape).astype(np.float32)

    builder = popart.Builder()

    lhs = builder.addInputTensor(popart.TensorInfo("FLOAT", lhs_shape), "lhs")
    rhs = builder.addInputTensor(popart.TensorInfo("FLOAT", rhs_shape), "rhs")

    o = builder.aiOnnx.matmul([lhs, rhs])
    builder.setSerializeMatMul({o}, "output_channels", 3)

    builder.addOutputTensor(o)

    proto = builder.getModelProto()

    dataFlow = popart.DataFlow(1, {o: popart.AnchorReturnType("ALL")})

    opts = getBaseOptions()

    pat = popart.Patterns(['MatMulOp', 'MatMulRhsGradOp', 'MatMulLhsGradOp'])

    with pytest.raises(popart.popart_exception) as e_info:
        session = popart.InferenceSession(
            fnModel=proto,
            dataFeed=dataFlow,
            userOptions=opts,
            passes=pat,
            deviceInfo=tu.get_ipu_model(compileIPUCode=False))

    assert (e_info.value.args[0].startswith(
        "Invalid serialisation factor 3 for output channels dim 4. output_channels dim should be a multple of the serialisation factor"
    ))


def test_matmul_serialization_inference(tmpdir):

    input_channels = 2
    reducing_dim = 2
    output_channels = 4

    lhs_shape = [input_channels, reducing_dim]
    rhs_shape = [reducing_dim, output_channels]
    lhs_data = np.random.rand(*lhs_shape).astype(np.float32)
    rhs_data = np.random.rand(*rhs_shape).astype(np.float32)

    def run_test(matmul_serialization_mode, matmul_serialization_factor,
                 verify):
        builder = popart.Builder()

        lhs = builder.addInputTensor(popart.TensorInfo("FLOAT", lhs_shape),
                                     "lhs")
        rhs = builder.addInputTensor(popart.TensorInfo("FLOAT", rhs_shape),
                                     "rhs")

        o = builder.aiOnnx.matmul([lhs, rhs])
        builder.setSerializeMatMul({o}, matmul_serialization_mode,
                                   matmul_serialization_factor)

        builder.addOutputTensor(o)

        proto = builder.getModelProto()

        dataFlow = popart.DataFlow(1, {o: popart.AnchorReturnType("ALL")})

        opts = getBaseOptions()

        pat = popart.Patterns(
            ['MatMulOp', 'MatMulRhsGradOp', 'MatMulLhsGradOp'])

        session = popart.InferenceSession(
            fnModel=proto,
            dataFeed=dataFlow,
            userOptions=opts,
            passes=pat,
            deviceInfo=tu.get_ipu_model(compileIPUCode=False))

        session.prepareDevice()

        anchors = session.initAnchorArrays()

        inputs = {lhs: lhs_data, rhs: rhs_data}
        stepio = popart.PyStepIO(inputs, anchors)

        session.run(stepio)

        verify(session, matmul_serialization_factor)

        return anchors[o]

    def verify_no_serialisation(session, matmul_serialization_factor):
        ''' Verify the the matmul in the main graphs is correct'''
        ir = json.loads(session._serializeIr(
            popart.IrSerializationFormat.JSON))
        matmuls = [op for op in ir['maingraph'] if op['type'] == 'MatMul']
        assert (len(matmuls) == 1)

        # forward
        lhs = matmuls[0]['inputs'][0]
        rhs = matmuls[0]['inputs'][1]
        assert (lhs['shape'] == gen_shape([1, input_channels, reducing_dim])
                and rhs['shape'] == gen_shape(
                    [1, reducing_dim, output_channels]))

    def verify_serialisation_input_channels(session,
                                            matmul_serialization_factor):
        ''' Verify the the matmul has the input sliced and is in a subgraph'''
        ir = json.loads(session._serializeIr(
            popart.IrSerializationFormat.JSON))
        matmuls = [op for op in ir['_subgraph(0)'] if op['type'] == 'MatMul']
        assert (len(matmuls) == 1)

        # forward
        lhs = matmuls[0]['inputs'][0]
        rhs = matmuls[0]['inputs'][1]
        assert (lhs['shape'] == gen_shape([
            1, input_channels // matmul_serialization_factor, reducing_dim
        ]) and rhs['shape'] == gen_shape([1, reducing_dim, output_channels]))

    def verify_serialisation_reducing_dim(session,
                                          matmul_serialization_factor):
        ''' Verify the the matmul has the input sliced and is in a subgraph'''
        ir = json.loads(session._serializeIr(
            popart.IrSerializationFormat.JSON))
        matmuls = [op for op in ir['_subgraph(0)'] if op['type'] == 'MatMul']
        assert (len(matmuls) == 1)

        # forward
        lhs = matmuls[0]['inputs'][0]
        rhs = matmuls[0]['inputs'][1]
        assert (lhs['shape'] == gen_shape([
            1, input_channels, reducing_dim // matmul_serialization_factor
        ]) and rhs['shape'] == gen_shape(
            [1, reducing_dim // matmul_serialization_factor, output_channels]))

    def verify_serialisation_output_channels(session,
                                             matmul_serialization_factor):
        ''' Verify the the matmul has the input sliced and is in a subgraph'''
        ir = json.loads(session._serializeIr(
            popart.IrSerializationFormat.JSON))
        matmuls = [op for op in ir['_subgraph(0)'] if op['type'] == 'MatMul']
        assert (len(matmuls) == 1)

        # forward
        lhs = matmuls[0]['inputs'][0]
        rhs = matmuls[0]['inputs'][1]
        assert (lhs['shape'] == gen_shape(
            [1, input_channels, reducing_dim]) and rhs['shape'] == gen_shape([
                1, reducing_dim, output_channels // matmul_serialization_factor
            ]))

    o1 = run_test("none", 0, verify_no_serialisation)
    o2 = run_test("input_channels", 2, verify_serialisation_input_channels)
    o3 = run_test("reducing_dim", 2, verify_serialisation_reducing_dim)
    o4 = run_test("output_channels", 4, verify_serialisation_output_channels)

    assert (np.allclose(o1, o2))
    assert (np.allclose(o1, o3))
    assert (np.allclose(o1, o4))


def test_matmul_serialization_training_1(tmpdir):

    input_channels = 6
    reducing_dim = 2
    output_channels = 4

    lhs_shape = [input_channels, reducing_dim]
    rhs_shape = [reducing_dim, output_channels]
    lhs_data = np.ones((*lhs_shape, ), dtype=np.float32)
    rhs_data = np.ones((*rhs_shape, ), dtype=np.float32)

    def run_test(matmul_serialization_mode, matmul_serialization_factor,
                 verify):
        builder = popart.Builder()

        lhs = builder.addInitializedInputTensor(lhs_data, "lhs")
        rhs = builder.addInitializedInputTensor(rhs_data, "rhs")

        o = builder.aiOnnx.matmul([lhs, rhs])

        builder.setSerializeMatMul({o}, matmul_serialization_mode,
                                   matmul_serialization_factor)

        proto = builder.getModelProto()

        dataFlow = popart.DataFlow(
            1,
            {
                o:
                popart.AnchorReturnType("ALL"),
                rhs:
                popart.AnchorReturnType("FINAL"),
                popart.reservedGradientPrefix() + lhs:
                popart.AnchorReturnType("ALL"),
                #popart.reservedGradientPrefix() + rhs: popart.AnchorReturnType("ALL"), << T11469
            })

        opts = getBaseOptions()

        pat = popart.Patterns(
            ['MatMulOp', 'MatMulRhsGradOp', 'MatMulLhsGradOp'])

        session = popart.TrainingSession(
            fnModel=proto,
            dataFeed=dataFlow,
            userOptions=opts,
            losses=[popart.L1Loss(o, "l1LossVal", 0.1)],
            optimizer=popart.ConstSGD(0.01),
            passes=pat,
            deviceInfo=tu.get_ipu_model(compileIPUCode=False))

        session.prepareDevice()

        session.weightsFromHost()
        session.optimizerFromHost()

        anchors = session.initAnchorArrays()

        inputs = {lhs: lhs_data}
        stepio = popart.PyStepIO(inputs, anchors)

        session.run(stepio)
        session.weightsToHost()

        verify(session, matmul_serialization_factor)

        return anchors[rhs]

    def verify_no_serialisation(session, matmul_serialization_factor):
        ''' Verify the the matmul in the main graphs is correct'''
        ir = json.loads(session._serializeIr(
            popart.IrSerializationFormat.JSON))
        matmuls = [op for op in ir['maingraph'] if op['type'] == 'MatMul']

        assert (len(matmuls) == 3)

        # forward
        lhs = matmuls[0]['inputs'][0]
        rhs = matmuls[0]['inputs'][1]
        assert (lhs['shape'] == gen_shape([1, input_channels, reducing_dim])
                and rhs['shape'] == gen_shape(
                    [1, reducing_dim, output_channels]))

        # bwd lhs
        bwd_lhs = _get_bwd_lhs_matmul(matmuls)
        lhs = bwd_lhs['inputs'][0]
        rhs = bwd_lhs['inputs'][1]
        assert (lhs['shape'] == gen_shape([1, input_channels, output_channels])
                and rhs['shape'] == gen_shape(
                    [1, output_channels, reducing_dim]))

        # bwd rhs
        bwd_rhs = _get_bwd_rhs_matmul(matmuls)
        lhs = bwd_rhs['inputs'][0]
        rhs = bwd_rhs['inputs'][1]
        assert (lhs['shape'] == gen_shape([1, reducing_dim, input_channels])
                and rhs['shape'] == gen_shape(
                    [1, input_channels, output_channels]))

    def verify_serialisation_input_channels(session,
                                            matmul_serialization_factor):
        ''' Verify the the matmul has the input sliced and is in a subgraph'''
        ir = json.loads(session._serializeIr(
            popart.IrSerializationFormat.JSON))

        matmuls = [op for op in ir['maingraph'] if op['type'] == 'MatMul']
        assert (len(matmuls) == 0)

        # FWD
        matmuls = [op for op in ir['_subgraph(0)'] if op['type'] == 'MatMul']
        assert (len(matmuls) == 1)

        lhs = matmuls[0]['inputs'][0]
        rhs = matmuls[0]['inputs'][1]

        assert (lhs['shape'] == gen_shape([
            1, input_channels // matmul_serialization_factor, reducing_dim
        ]) and rhs['shape'] == gen_shape([1, reducing_dim, output_channels]))

        # BWD_LHS
        matmuls = [op for op in ir['_subgraph(2)'] if op['type'] == 'MatMul']
        assert (len(matmuls) == 1)

        lhs = matmuls[0]['inputs'][0]
        rhs = matmuls[0]['inputs'][1]

        assert (lhs['shape'] == gen_shape([
            1, input_channels // matmul_serialization_factor, output_channels
        ]) and rhs['shape'] == gen_shape([1, output_channels, reducing_dim]))

        # BWD_RHS
        matmuls = [op for op in ir['_subgraph(1)'] if op['type'] == 'MatMul']
        assert (len(matmuls) == 1)

        lhs = matmuls[0]['inputs'][0]
        rhs = matmuls[0]['inputs'][1]

        assert (lhs['shape'] == gen_shape([
            1, reducing_dim, input_channels // matmul_serialization_factor
        ]) and rhs['shape'] == gen_shape([
            1, input_channels // matmul_serialization_factor, output_channels
        ]))

    def verify_serialisation_reducing_dim(session,
                                          matmul_serialization_factor):
        ''' Verify the the matmul has the input sliced and is in a subgraph'''
        ir = json.loads(session._serializeIr(
            popart.IrSerializationFormat.JSON))

        matmuls = [op for op in ir['maingraph'] if op['type'] == 'MatMul']
        assert (len(matmuls) == 0)

        # FWD
        matmuls = [op for op in ir['_subgraph(0)'] if op['type'] == 'MatMul']
        assert (len(matmuls) == 1)

        lhs = matmuls[0]['inputs'][0]
        rhs = matmuls[0]['inputs'][1]

        assert (lhs['shape'] == gen_shape([
            1, input_channels, reducing_dim // matmul_serialization_factor
        ]) and rhs['shape'] == gen_shape(
            [1, reducing_dim // matmul_serialization_factor, output_channels]))

        # BWD_LHS
        matmuls = [op for op in ir['_subgraph(2)'] if op['type'] == 'MatMul']
        assert (len(matmuls) == 1)

        lhs = matmuls[0]['inputs'][0]
        rhs = matmuls[0]['inputs'][1]

        assert (lhs['shape'] == gen_shape([1, input_channels, output_channels])
                and rhs['shape'] == gen_shape([
                    1, output_channels,
                    reducing_dim // matmul_serialization_factor
                ]))

        # BWD_RHS
        matmuls = [op for op in ir['_subgraph(1)'] if op['type'] == 'MatMul']
        assert (len(matmuls) == 1)

        lhs = matmuls[0]['inputs'][0]
        rhs = matmuls[0]['inputs'][1]

        assert (lhs['shape'] == gen_shape([
            1, reducing_dim // matmul_serialization_factor, input_channels
        ]) and rhs['shape'] == gen_shape([1, input_channels, output_channels]))

    def verify_serialisation_output_channels(session,
                                             matmul_serialization_factor):
        ''' Verify the the matmul has the input sliced and is in a subgraph'''
        ir = json.loads(session._serializeIr(
            popart.IrSerializationFormat.JSON))

        matmuls = [op for op in ir['maingraph'] if op['type'] == 'MatMul']
        assert (len(matmuls) == 0)

        # FWD
        matmuls = [op for op in ir['_subgraph(0)'] if op['type'] == 'MatMul']
        assert (len(matmuls) == 1)

        lhs = matmuls[0]['inputs'][0]
        rhs = matmuls[0]['inputs'][1]

        assert (lhs['shape'] == gen_shape(
            [1, input_channels, reducing_dim]) and rhs['shape'] == gen_shape([
                1, reducing_dim, output_channels // matmul_serialization_factor
            ]))

        # BWD_LHS
        matmuls = [op for op in ir['_subgraph(2)'] if op['type'] == 'MatMul']
        assert (len(matmuls) == 1)

        lhs = matmuls[0]['inputs'][0]
        rhs = matmuls[0]['inputs'][1]

        assert (lhs['shape'] == gen_shape([
            1, input_channels, output_channels // matmul_serialization_factor
        ]) and rhs['shape'] == gen_shape(
            [1, output_channels // matmul_serialization_factor, reducing_dim]))

        # BWD_RHS
        matmuls = [op for op in ir['_subgraph(1)'] if op['type'] == 'MatMul']
        assert (len(matmuls) == 1)

        lhs = matmuls[0]['inputs'][0]
        rhs = matmuls[0]['inputs'][1]

        assert (lhs['shape'] == gen_shape([1, reducing_dim, input_channels])
                and rhs['shape'] == gen_shape([
                    1, input_channels,
                    output_channels // matmul_serialization_factor
                ]))

    w1 = run_test("none", 0, verify_no_serialisation)
    w2 = run_test("input_channels", 2, verify_serialisation_input_channels)
    w3 = run_test("reducing_dim", 2, verify_serialisation_reducing_dim)
    w4 = run_test("output_channels", 4, verify_serialisation_output_channels)

    assert (np.allclose(w1, w2))
    assert (np.allclose(w1, w3))
    assert (np.allclose(w1, w4))


def test_matmul_serialization_training_2(tmpdir):

    input_channels = 6
    reducing_dim = 16
    output_channels = 15

    lhs_group_dim = 2

    lhs_shape = [lhs_group_dim, input_channels, reducing_dim]
    rhs_shape = [reducing_dim, output_channels]
    lhs_data = np.ones((*lhs_shape, ), dtype=np.float32)
    rhs_data = np.ones((*rhs_shape, ), dtype=np.float32)

    def run_test(matmul_serialization_mode, matmul_serialization_factor,
                 verify):
        builder = popart.Builder()

        lhs = builder.addInputTensor(popart.TensorInfo("FLOAT", lhs_shape),
                                     "lhs")

        lhs_reshape = builder.reshape_const(
            builder.aiOnnx, [lhs],
            [lhs_group_dim * input_channels, reducing_dim])
        rhs = builder.addInitializedInputTensor(rhs_data, "rhs")

        o = builder.aiOnnx.matmul([lhs_reshape, rhs])

        builder.setSerializeMatMul({o}, matmul_serialization_mode,
                                   matmul_serialization_factor)

        o_reshape = builder.reshape_const(
            builder.aiOnnx, [o],
            [lhs_group_dim, input_channels, output_channels])

        proto = builder.getModelProto()

        dataFlow = popart.DataFlow(
            1, {
                o_reshape:
                popart.AnchorReturnType("ALL"),
                rhs:
                popart.AnchorReturnType("FINAL"),
                popart.reservedGradientPrefix() + lhs:
                popart.AnchorReturnType("ALL"),
            })

        opts = getBaseOptions()

        pat = popart.Patterns(
            ['MatMulOp', 'MatMulRhsGradOp', 'MatMulLhsGradOp'])

        session = popart.TrainingSession(
            fnModel=proto,
            dataFeed=dataFlow,
            userOptions=opts,
            losses=[popart.L1Loss(o, "l1LossVal", 0.1)],
            optimizer=popart.ConstSGD(0.01),
            passes=pat,
            deviceInfo=tu.get_ipu_model(compileIPUCode=False))

        session.prepareDevice()

        session.weightsFromHost()
        session.optimizerFromHost()

        anchors = session.initAnchorArrays()

        inputs = {lhs: lhs_data}
        stepio = popart.PyStepIO(inputs, anchors)

        session.run(stepio)

        session.weightsToHost()
        verify(session, matmul_serialization_factor)

        return anchors[rhs]

    def verify_no_serialisation(session, matmul_serialization_factor):
        ''' Verify the the matmul in the main graphs is correct'''
        ir = json.loads(session._serializeIr(
            popart.IrSerializationFormat.JSON))
        matmuls = [op for op in ir['maingraph'] if op['type'] == 'MatMul']

        assert (len(matmuls) == 3)

        # forward
        lhs = matmuls[0]['inputs'][0]
        rhs = matmuls[0]['inputs'][1]
        assert (lhs['shape'] == gen_shape([
            1, lhs_group_dim * input_channels, reducing_dim
        ]) and rhs['shape'] == gen_shape([1, reducing_dim, output_channels]))

        # bwd lhs
        bwd_lhs = _get_bwd_lhs_matmul(matmuls)
        lhs = bwd_lhs['inputs'][0]
        rhs = bwd_lhs['inputs'][1]
        assert (lhs['shape'] == gen_shape([
            1, lhs_group_dim * input_channels, output_channels
        ]) and rhs['shape'] == gen_shape([1, output_channels, reducing_dim]))

        # bwd rhs
        bwd_rhs = _get_bwd_rhs_matmul(matmuls)
        lhs = bwd_rhs['inputs'][0]
        rhs = bwd_rhs['inputs'][1]

        assert (lhs['shape'] == gen_shape(
            [1, reducing_dim, lhs_group_dim * input_channels])
                and rhs['shape'] == gen_shape(
                    [1, lhs_group_dim * input_channels, output_channels]))

    def verify_serialisation_input_channels(session,
                                            matmul_serialization_factor):
        ''' Verify the the matmul has the input sliced and is in a subgraph'''
        ir = json.loads(session._serializeIr(
            popart.IrSerializationFormat.JSON))

        matmuls = [op for op in ir['maingraph'] if op['type'] == 'MatMul']
        assert (len(matmuls) == 0)

        # FWD
        matmuls = [op for op in ir['_subgraph(0)'] if op['type'] == 'MatMul']
        assert (len(matmuls) == 1)

        lhs = matmuls[0]['inputs'][0]
        rhs = matmuls[0]['inputs'][1]

        assert (lhs['shape'] == gen_shape([
            1, (lhs_group_dim * input_channels) // matmul_serialization_factor,
            reducing_dim
        ]) and rhs['shape'] == gen_shape([1, reducing_dim, output_channels]))

        # BWD_LHS
        matmuls = [op for op in ir['_subgraph(2)'] if op['type'] == 'MatMul']
        assert (len(matmuls) == 1)

        lhs = matmuls[0]['inputs'][0]
        rhs = matmuls[0]['inputs'][1]

        assert (lhs['shape'] == gen_shape([
            1, (lhs_group_dim * input_channels) // matmul_serialization_factor,
            output_channels
        ]) and rhs['shape'] == gen_shape([1, output_channels, reducing_dim]))

        # BWD_RHS
        matmuls = [op for op in ir['_subgraph(1)'] if op['type'] == 'MatMul']
        assert (len(matmuls) == 1)

        lhs = matmuls[0]['inputs'][0]
        rhs = matmuls[0]['inputs'][1]

        assert (lhs['shape'] == gen_shape([
            1, reducing_dim,
            (lhs_group_dim * input_channels) // matmul_serialization_factor
        ]) and rhs['shape'] == gen_shape([
            1, (lhs_group_dim * input_channels) // matmul_serialization_factor,
            output_channels
        ]))

    def verify_serialisation_reducing_dim(session,
                                          matmul_serialization_factor):
        ''' Verify the the matmul has the input sliced and is in a subgraph'''
        ir = json.loads(session._serializeIr(
            popart.IrSerializationFormat.JSON))

        matmuls = [op for op in ir['maingraph'] if op['type'] == 'MatMul']
        assert (len(matmuls) == 0)

        # FWD
        matmuls = [op for op in ir['_subgraph(0)'] if op['type'] == 'MatMul']
        assert (len(matmuls) == 1)

        lhs = matmuls[0]['inputs'][0]
        rhs = matmuls[0]['inputs'][1]

        assert (lhs['shape'] == gen_shape([
            1, (lhs_group_dim * input_channels),
            reducing_dim // matmul_serialization_factor
        ]) and rhs['shape'] == gen_shape(
            [1, reducing_dim // matmul_serialization_factor, output_channels]))

        # BWD_LHS
        matmuls = [op for op in ir['_subgraph(2)'] if op['type'] == 'MatMul']
        assert (len(matmuls) == 1)

        lhs = matmuls[0]['inputs'][0]
        rhs = matmuls[0]['inputs'][1]

        assert (lhs['shape'] == gen_shape([
            1, (lhs_group_dim * input_channels), output_channels
        ]) and rhs['shape'] == gen_shape(
            [1, output_channels, reducing_dim // matmul_serialization_factor]))

        # BWD_RHS
        matmuls = [op for op in ir['_subgraph(1)'] if op['type'] == 'MatMul']
        assert (len(matmuls) == 1)

        lhs = matmuls[0]['inputs'][0]
        rhs = matmuls[0]['inputs'][1]

        assert (lhs['shape'] == gen_shape([
            1, reducing_dim // matmul_serialization_factor,
            (lhs_group_dim * input_channels)
        ]) and rhs['shape'] == gen_shape(
            [1, (lhs_group_dim * input_channels), output_channels]))

    def verify_serialisation_output_channels(session,
                                             matmul_serialization_factor):
        ''' Verify the the matmul has the input sliced and is in a subgraph'''
        ir = json.loads(session._serializeIr(
            popart.IrSerializationFormat.JSON))

        matmuls = [op for op in ir['maingraph'] if op['type'] == 'MatMul']
        assert (len(matmuls) == 0)

        # FWD
        matmuls = [op for op in ir['_subgraph(0)'] if op['type'] == 'MatMul']
        assert (len(matmuls) == 1)

        lhs = matmuls[0]['inputs'][0]
        rhs = matmuls[0]['inputs'][1]

        assert (lhs['shape'] == gen_shape([
            1, (lhs_group_dim * input_channels), reducing_dim
        ]) and rhs['shape'] == gen_shape(
            [1, reducing_dim, output_channels // matmul_serialization_factor]))

        # BWD_LHS
        matmuls = [op for op in ir['_subgraph(2)'] if op['type'] == 'MatMul']
        assert (len(matmuls) == 1)

        lhs = matmuls[0]['inputs'][0]
        rhs = matmuls[0]['inputs'][1]

        assert (lhs['shape'] == gen_shape([
            1, (lhs_group_dim * input_channels),
            output_channels // matmul_serialization_factor
        ]) and rhs['shape'] == gen_shape(
            [1, output_channels // matmul_serialization_factor, reducing_dim]))

        # BWD_RHS
        matmuls = [op for op in ir['_subgraph(1)'] if op['type'] == 'MatMul']
        assert (len(matmuls) == 1)

        lhs = matmuls[0]['inputs'][0]
        rhs = matmuls[0]['inputs'][1]

        assert (lhs['shape'] == gen_shape(
            [1, reducing_dim, (lhs_group_dim * input_channels)])
                and rhs['shape'] == gen_shape([
                    1, (lhs_group_dim * input_channels),
                    output_channels // matmul_serialization_factor
                ]))

    w1 = run_test("none", 0, verify_no_serialisation)
    w2 = run_test("input_channels", 2, verify_serialisation_input_channels)
    w3 = run_test("reducing_dim", 4, verify_serialisation_reducing_dim)
    w4 = run_test("output_channels", 5, verify_serialisation_output_channels)

    assert (np.allclose(w1, w2))
    assert (np.allclose(w1, w3))
    assert (np.allclose(w1, w4))


def test_matmul_serialization_precision(tmpdir):
    np.random.seed(1984)

    input_channels = 20
    reducing_dim = 128
    output_channels = 64

    lhs_shape = [input_channels, reducing_dim]
    rhs_shape = [reducing_dim, output_channels]
    lhs_data = np.random.normal(0, 0.02, (*lhs_shape, )).astype(np.float16)
    rhs_data = np.random.normal(0, 0.02, (*rhs_shape, )).astype(np.float16)

    def run_test(matmul_serialization_mode, matmul_serialization_factor,
                 verify):
        builder = popart.Builder()

        lhs = builder.addInitializedInputTensor(lhs_data, "lhs")
        rhs = builder.addInitializedInputTensor(rhs_data, "rhs")

        o = builder.aiOnnx.matmul([lhs, rhs])

        builder.setSerializeMatMul({o},
                                   matmul_serialization_mode,
                                   matmul_serialization_factor,
                                   keep_precision=True)

        proto = builder.getModelProto()

        dataFlow = popart.DataFlow(
            1,
            {
                o:
                popart.AnchorReturnType("ALL"),
                rhs:
                popart.AnchorReturnType("FINAL"),
                popart.reservedGradientPrefix() + lhs:
                popart.AnchorReturnType("ALL"),
                #popart.reservedGradientPrefix() + rhs: popart.AnchorReturnType("ALL"), << T11469
            })

        opts = getBaseOptions()

        pat = popart.Patterns(
            ['MatMulOp', 'MatMulRhsGradOp', 'MatMulLhsGradOp'])

        session = popart.TrainingSession(
            fnModel=proto,
            dataFeed=dataFlow,
            userOptions=opts,
            losses=[popart.L1Loss(o, "l1LossVal", 0.1)],
            optimizer=popart.ConstSGD(0.01),
            passes=pat,
            deviceInfo=tu.get_ipu_model(compileIPUCode=False))

        session.prepareDevice()

        session.weightsFromHost()
        session.optimizerFromHost()

        anchors = session.initAnchorArrays()

        inputs = {lhs: lhs_data}
        stepio = popart.PyStepIO(inputs, anchors)

        session.run(stepio)
        session.weightsToHost()

        verify(session, matmul_serialization_factor)

        return anchors[rhs]

    def verify_no_serialisation(session, matmul_serialization_factor):
        ''' Verify the the matmul in the main graphs is correct'''
        ir = json.loads(session._serializeIr(
            popart.IrSerializationFormat.JSON))
        matmuls = [op for op in ir['maingraph'] if op['type'] == 'MatMul']

        assert (len(matmuls) == 3)

        # forward
        lhs = matmuls[0]['inputs'][0]
        rhs = matmuls[0]['inputs'][1]
        assert (lhs['shape'] == gen_shape([1, input_channels, reducing_dim])
                and rhs['shape'] == gen_shape(
                    [1, reducing_dim, output_channels]))

        # bwd lhs
        bwd_lhs = _get_bwd_lhs_matmul(matmuls)
        lhs = bwd_lhs['inputs'][0]
        rhs = bwd_lhs['inputs'][1]
        assert (lhs['shape'] == gen_shape([1, input_channels, output_channels])
                and rhs['shape'] == gen_shape(
                    [1, output_channels, reducing_dim]))

        # bwd rhs
        bwd_rhs = _get_bwd_rhs_matmul(matmuls)
        lhs = bwd_rhs['inputs'][0]
        rhs = bwd_rhs['inputs'][1]
        assert (lhs['shape'] == gen_shape([1, reducing_dim, input_channels])
                and rhs['shape'] == gen_shape(
                    [1, input_channels, output_channels]))

    def verify_serialisation_input_channels(session,
                                            matmul_serialization_factor):
        ''' Verify the the matmul has the input sliced and is in a subgraph'''
        ir = json.loads(session._serializeIr(
            popart.IrSerializationFormat.JSON))

        casts = [op for op in ir['maingraph'] if op['type'] == 'Cast']
        assert (len(casts) == 1)

        matmuls = [op for op in ir['maingraph'] if op['type'] == 'MatMul']
        assert (len(matmuls) == 0)

        # FWD
        matmuls = [op for op in ir['_subgraph(0)'] if op['type'] == 'MatMul']
        assert (len(matmuls) == 1)

        lhs = matmuls[0]['inputs'][0]
        rhs = matmuls[0]['inputs'][1]

        assert (lhs['shape'] == gen_shape([
            1, input_channels // matmul_serialization_factor, reducing_dim
        ]) and rhs['shape'] == gen_shape([1, reducing_dim, output_channels]))

        # BWD_LHS
        matmuls = [op for op in ir['_subgraph(2)'] if op['type'] == 'MatMul']
        assert (len(matmuls) == 1)

        lhs = matmuls[0]['inputs'][0]
        rhs = matmuls[0]['inputs'][1]

        assert (lhs['shape'] == gen_shape([
            1, input_channels // matmul_serialization_factor, output_channels
        ]) and rhs['shape'] == gen_shape([1, output_channels, reducing_dim]))

        # BWD_RHS
        matmuls = [op for op in ir['_subgraph(1)'] if op['type'] == 'MatMul']
        assert (len(matmuls) == 1)

        lhs = matmuls[0]['inputs'][0]
        rhs = matmuls[0]['inputs'][1]

        assert (lhs['shape'] == gen_shape([
            1, reducing_dim, input_channels // matmul_serialization_factor
        ]) and rhs['shape'] == gen_shape([
            1, input_channels // matmul_serialization_factor, output_channels
        ]))

    def verify_serialisation_reducing_dim(session,
                                          matmul_serialization_factor):
        ''' Verify the the matmul has the input sliced and is in a subgraph'''
        ir = json.loads(session._serializeIr(
            popart.IrSerializationFormat.JSON))

        casts = [op for op in ir['maingraph'] if op['type'] == 'Cast']
        assert (len(casts) == 1)

        matmuls = [op for op in ir['maingraph'] if op['type'] == 'MatMul']
        assert (len(matmuls) == 0)

        # FWD
        matmuls = [op for op in ir['_subgraph(0)'] if op['type'] == 'MatMul']
        assert (len(matmuls) == 1)

        lhs = matmuls[0]['inputs'][0]
        rhs = matmuls[0]['inputs'][1]

        assert (lhs['shape'] == gen_shape(
            [1, input_channels, reducing_dim // matmul_serialization_factor]))
        assert (rhs['shape'] == gen_shape(
            [1, reducing_dim // matmul_serialization_factor, output_channels]))

        # BWD_LHS
        matmuls = [op for op in ir['_subgraph(2)'] if op['type'] == 'MatMul']
        assert (len(matmuls) == 1)

        lhs = matmuls[0]['inputs'][0]
        rhs = matmuls[0]['inputs'][1]

        assert (lhs['shape'] == gen_shape([1, input_channels, output_channels])
                and rhs['shape'] == gen_shape([
                    1, output_channels,
                    reducing_dim // matmul_serialization_factor
                ]))

        # BWD_RHS
        matmuls = [op for op in ir['_subgraph(1)'] if op['type'] == 'MatMul']
        assert (len(matmuls) == 1)

        lhs = matmuls[0]['inputs'][0]
        rhs = matmuls[0]['inputs'][1]

        assert (lhs['shape'] == gen_shape([
            1, reducing_dim // matmul_serialization_factor, input_channels
        ]) and rhs['shape'] == gen_shape([1, input_channels, output_channels]))

    def verify_serialisation_output_channels(session,
                                             matmul_serialization_factor):
        ''' Verify the the matmul has the input sliced and is in a subgraph'''
        ir = json.loads(session._serializeIr(
            popart.IrSerializationFormat.JSON))

        casts = [op for op in ir['maingraph'] if op['type'] == 'Cast']
        assert (len(casts) == 1)

        matmuls = [op for op in ir['maingraph'] if op['type'] == 'MatMul']
        assert (len(matmuls) == 0)

        # FWD
        matmuls = [op for op in ir['_subgraph(0)'] if op['type'] == 'MatMul']
        assert (len(matmuls) == 1)

        lhs = matmuls[0]['inputs'][0]
        rhs = matmuls[0]['inputs'][1]

        assert (lhs['shape'] == gen_shape(
            [1, input_channels, reducing_dim]) and rhs['shape'] == gen_shape([
                1, reducing_dim, output_channels // matmul_serialization_factor
            ]))

        # BWD_LHS
        matmuls = [op for op in ir['_subgraph(2)'] if op['type'] == 'MatMul']
        assert (len(matmuls) == 1)

        lhs = matmuls[0]['inputs'][0]
        rhs = matmuls[0]['inputs'][1]

        assert (lhs['shape'] == gen_shape([
            1, input_channels, output_channels // matmul_serialization_factor
        ]) and rhs['shape'] == gen_shape(
            [1, output_channels // matmul_serialization_factor, reducing_dim]))

        # BWD_RHS
        matmuls = [op for op in ir['_subgraph(1)'] if op['type'] == 'MatMul']
        assert (len(matmuls) == 1)

        lhs = matmuls[0]['inputs'][0]
        rhs = matmuls[0]['inputs'][1]

        assert (lhs['shape'] == gen_shape([1, reducing_dim, input_channels])
                and rhs['shape'] == gen_shape([
                    1, input_channels,
                    output_channels // matmul_serialization_factor
                ]))

    w1 = run_test("none", 0, verify_no_serialisation)
    w2 = run_test("input_channels", 10, verify_serialisation_input_channels)
    w3 = run_test("reducing_dim", 16, verify_serialisation_reducing_dim)
    w4 = run_test("output_channels", 16, verify_serialisation_output_channels)

    # Check to make sure that the output (9)anchors[rhs]) of the test is the same without
    # serialization and with different serialization modes.

    assert (np.allclose(w1, w2))
    assert (np.allclose(w1, w3))
    assert (np.allclose(w1, w4))


def test_matmul_serialization_training_with_gradient_accumlation(tmpdir):

    input_channels = 6
    reducing_dim = 2
    output_channels = 4

    batches_per_step = 20

    lhs_shape = [input_channels, reducing_dim]
    rhs_shape = [reducing_dim, output_channels]
    lhs_data = np.ones((
        batches_per_step,
        *lhs_shape,
    ), dtype=np.float32)
    rhs_data = np.ones((
        batches_per_step,
        *rhs_shape,
    ), dtype=np.float32)

    def run_test(matmul_serialization_mode, matmul_serialization_factor,
                 verify):
        builder = popart.Builder()

        lhs = builder.addInitializedInputTensor(lhs_data, "lhs")
        rhs = builder.addInitializedInputTensor(rhs_data, "rhs")

        o = builder.aiOnnx.matmul([lhs, rhs])

        builder.setSerializeMatMul({o}, matmul_serialization_mode,
                                   matmul_serialization_factor)

        proto = builder.getModelProto()

        dataFlow = popart.DataFlow(
            20,
            {
                o:
                popart.AnchorReturnType("ALL"),
                rhs:
                popart.AnchorReturnType("FINAL"),
                popart.reservedGradientPrefix() + lhs:
                popart.AnchorReturnType("ALL"),
                #popart.reservedGradientPrefix() + rhs: popart.AnchorReturnType("ALL"), << T11469
            })

        opts = getBaseOptions()

        opts.enableGradientAccumulation = True
        opts.accumulationFactor = 5

        pat = popart.Patterns(
            ['MatMulOp', 'MatMulRhsGradOp', 'MatMulLhsGradOp'])

        session = popart.TrainingSession(
            fnModel=proto,
            dataFeed=dataFlow,
            userOptions=opts,
            losses=[popart.L1Loss(o, "l1LossVal", 0.1)],
            optimizer=popart.ConstSGD(0.01),
            passes=pat,
            deviceInfo=tu.get_ipu_model(compileIPUCode=False))

        session.prepareDevice()

        session.weightsFromHost()
        session.optimizerFromHost()

        anchors = session.initAnchorArrays()

        inputs = {lhs: lhs_data}
        stepio = popart.PyStepIO(inputs, anchors)

        #TODO (T15448, understand why shapes are incorrect)
        stepio.enableRuntimeAsserts(False)

        session.run(stepio)
        session.weightsToHost()

        verify(session, matmul_serialization_factor)

        return anchors[rhs]

    def verify_no_serialisation(session, matmul_serialization_factor):
        ''' Verify the the matmul in the main graphs is correct'''
        ir = json.loads(session._serializeIr(
            popart.IrSerializationFormat.JSON))
        matmuls = [op for op in ir['maingraph'] if op['type'] == 'MatMul']

        # With no serialization we should have 3 matmul's not outlined
        # FWD, BWD_LHS, BWD_RHS
        assert (len(matmuls) == 3)

        # forward
        lhs = matmuls[0]['inputs'][0]
        rhs = matmuls[0]['inputs'][1]
        assert (lhs['shape'] == gen_shape(
            [batches_per_step, input_channels, reducing_dim])
                and rhs['shape'] == gen_shape(
                    [batches_per_step, reducing_dim, output_channels]))

        # bwd lhs
        lhs = matmuls[2]['inputs'][0]
        rhs = matmuls[2]['inputs'][1]
        assert (lhs['shape'] == gen_shape(
            [batches_per_step, input_channels, output_channels])
                and rhs['shape'] == gen_shape(
                    [batches_per_step, output_channels, reducing_dim]))

        # bwd rhs
        lhs = matmuls[1]['inputs'][0]
        rhs = matmuls[1]['inputs'][1]
        assert (lhs['shape'] == gen_shape(
            [batches_per_step, reducing_dim, input_channels])
                and rhs['shape'] == gen_shape(
                    [batches_per_step, input_channels, output_channels]))

    def verify_serialisation_input_channels(session,
                                            matmul_serialization_factor):
        ''' Verify the the matmul has the input sliced and is in a subgraph'''
        ir = json.loads(session._serializeIr(
            popart.IrSerializationFormat.JSON))

        matmuls = [op for op in ir['maingraph'] if op['type'] == 'MatMul']
        assert (len(matmuls) == 0)

        # FWD
        matmuls = [op for op in ir['_subgraph(0)'] if op['type'] == 'MatMul']
        assert (len(matmuls) == 1)

        lhs = matmuls[0]['inputs'][0]
        rhs = matmuls[0]['inputs'][1]

        assert (lhs['shape'] == gen_shape([
            batches_per_step, input_channels // matmul_serialization_factor,
            reducing_dim
        ]) and rhs['shape'] == gen_shape(
            [batches_per_step, reducing_dim, output_channels]))

        # BWD_LHS
        matmuls = [op for op in ir['_subgraph(2)'] if op['type'] == 'MatMul']
        assert (len(matmuls) == 1)

        lhs = matmuls[0]['inputs'][0]
        rhs = matmuls[0]['inputs'][1]

        assert (lhs['shape'] == gen_shape([
            batches_per_step, input_channels // matmul_serialization_factor,
            output_channels
        ]) and rhs['shape'] == gen_shape(
            [batches_per_step, output_channels, reducing_dim]))

        # BWD_RHS
        matmuls = [op for op in ir['_subgraph(1)'] if op['type'] == 'MatMul']
        assert (len(matmuls) == 1)

        lhs = matmuls[0]['inputs'][0]
        rhs = matmuls[0]['inputs'][1]

        assert (lhs['shape'] == gen_shape([
            batches_per_step, reducing_dim,
            input_channels // matmul_serialization_factor
        ]) and rhs['shape'] == gen_shape([
            batches_per_step, input_channels // matmul_serialization_factor,
            output_channels
        ]))

    def verify_serialisation_reducing_dim(session,
                                          matmul_serialization_factor):
        ''' Verify the the matmul has the input sliced and is in a subgraph'''
        ir = json.loads(session._serializeIr(
            popart.IrSerializationFormat.JSON))
        print(ir)
        matmuls = [op for op in ir['maingraph'] if op['type'] == 'MatMul']
        assert (len(matmuls) == 0)

        # FWD
        matmuls = [op for op in ir['_subgraph(0)'] if op['type'] == 'MatMul']
        assert (len(matmuls) == 1)

        lhs = matmuls[0]['inputs'][0]
        rhs = matmuls[0]['inputs'][1]

        assert (lhs['shape'] == gen_shape([
            batches_per_step, input_channels,
            reducing_dim // matmul_serialization_factor
        ]) and rhs['shape'] == gen_shape([
            batches_per_step, reducing_dim // matmul_serialization_factor,
            output_channels
        ]))

        # BWD_LHS
        matmuls = [op for op in ir['_subgraph(2)'] if op['type'] == 'MatMul']
        assert (len(matmuls) == 1)

        lhs = matmuls[0]['inputs'][0]
        rhs = matmuls[0]['inputs'][1]

        assert (lhs['shape'] == gen_shape(
            [batches_per_step, input_channels, output_channels])
                and rhs['shape'] == gen_shape([
                    batches_per_step, output_channels,
                    reducing_dim // matmul_serialization_factor
                ]))

        # BWD_RHS
        matmuls = [op for op in ir['_subgraph(1)'] if op['type'] == 'MatMul']
        assert (len(matmuls) == 1)

        lhs = matmuls[0]['inputs'][0]
        rhs = matmuls[0]['inputs'][1]

        assert (lhs['shape'] == gen_shape([
            batches_per_step, reducing_dim // matmul_serialization_factor,
            input_channels
        ]) and rhs['shape'] == gen_shape(
            [batches_per_step, input_channels, output_channels]))

    def verify_serialisation_output_channels(session,
                                             matmul_serialization_factor):
        ''' Verify the the matmul has the input sliced and is in a subgraph'''
        ir = json.loads(session._serializeIr(
            popart.IrSerializationFormat.JSON))

        matmuls = [op for op in ir['maingraph'] if op['type'] == 'MatMul']
        assert (len(matmuls) == 0)

        # FWD
        matmuls = [op for op in ir['_subgraph(0)'] if op['type'] == 'MatMul']
        assert (len(matmuls) == 1)

        lhs = matmuls[0]['inputs'][0]
        rhs = matmuls[0]['inputs'][1]

        assert (lhs['shape'] == gen_shape(
            [batches_per_step, input_channels, reducing_dim])
                and rhs['shape'] == gen_shape([
                    batches_per_step, reducing_dim,
                    output_channels // matmul_serialization_factor
                ]))

        # BWD_LHS
        matmuls = [op for op in ir['_subgraph(2)'] if op['type'] == 'MatMul']
        assert (len(matmuls) == 1)

        lhs = matmuls[0]['inputs'][0]
        rhs = matmuls[0]['inputs'][1]

        assert (lhs['shape'] == gen_shape([
            batches_per_step, input_channels,
            output_channels // matmul_serialization_factor
        ]) and rhs['shape'] == gen_shape([
            batches_per_step, output_channels // matmul_serialization_factor,
            reducing_dim
        ]))

        # BWD_RHS
        matmuls = [op for op in ir['_subgraph(1)'] if op['type'] == 'MatMul']
        assert (len(matmuls) == 1)

        lhs = matmuls[0]['inputs'][0]
        rhs = matmuls[0]['inputs'][1]

        assert (lhs['shape'] == gen_shape(
            [batches_per_step, reducing_dim, input_channels])
                and rhs['shape'] == gen_shape([
                    batches_per_step, input_channels,
                    output_channels // matmul_serialization_factor
                ]))

    w1 = run_test("none", 0, verify_no_serialisation)
    w2 = run_test("input_channels", 2, verify_serialisation_input_channels)
    w3 = run_test("reducing_dim", 2, verify_serialisation_reducing_dim)
    w4 = run_test("output_channels", 4, verify_serialisation_output_channels)

    assert (np.allclose(w1, w2))
    assert (np.allclose(w1, w3))
    assert (np.allclose(w1, w4))
