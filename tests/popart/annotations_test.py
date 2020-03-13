# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import popart
import numpy as np
import json
import test_util as tu


def _get_ir(pingpong_enabled, virtualgraph_enabled, pipeline_enabled):
    dsize = 10
    builder = popart.Builder()
    ip = builder.addInputTensor(popart.TensorInfo("FLOAT", [dsize, dsize]))

    anchorIds = []
    anchorIds.append(popart.reservedGradientPrefix() + ip)

    def add_layer(in_id):
        w = builder.addInitializedInputTensor(
            np.ones([dsize, dsize], np.float32))
        matmul_id = builder.aiOnnx.matmul([in_id, w])
        return matmul_id

    x = ip
    for i in range(3):
        w = builder.addInitializedInputTensor(
            np.ones([dsize, dsize], np.float32))
        x = builder.aiOnnx.matmul([x, w])

        if pingpong_enabled:
            builder.pingPongPhase(x, i)
        if virtualgraph_enabled:
            builder.virtualGraph(x, i)
        if pipeline_enabled:
            builder.pipelineStage(x, i)

        anchorIds.append(popart.reservedGradientPrefix() + x)

    out = x
    builder.addOutputTensor(out)

    device = tu.create_test_device()

    dfAnchors = {}
    for anchorId in anchorIds:
        dfAnchors.update({anchorId: popart.AnchorReturnType("ALL")})

    opts = popart.SessionOptions()
    # disable outlining to make the ir easier to parse
    opts.enableOutlining = False

    proto = builder.getModelProto()

    loss = popart.L1Loss(out, 'l1LossVal', 0.1)
    if virtualgraph_enabled:
        loss.virtualGraph(3)

    session = popart.TrainingSession(fnModel=proto,
                                     dataFeed=popart.DataFlow(1, dfAnchors),
                                     optimizer=popart.ConstSGD(0.1),
                                     losses=[loss],
                                     passes=popart.Patterns(
                                         popart.PatternsLevel.ALL),
                                     userOptions=opts,
                                     deviceInfo=device)

    ir = json.loads(session._serializeIr(popart.IrSerializationFormat.JSON))
    return ir


def _compare_irs(ir1, ir2):
    # there should only be one graph in the ir
    assert len(ir1) == 1
    assert len(ir2) == 1

    g1 = ir1['maingraph']
    g2 = ir2['maingraph']

    # graphs should have the same number of ops
    assert len(g1) == len(g2)

    def get_op_comparison_string(op):
        name = op['name']
        op_type = op['type']
        if name:
            return f'{name}:{op_type}'
        else:
            return f'{op_type}'

    for op1, op2 in zip(g1, g2):
        op1 = get_op_comparison_string(op1)
        op2 = get_op_comparison_string(op2)
        print(f'{op1:<60} | {op2:<60} | {op1 == op2}')
        assert op1 == op2


# Check that the ping pong phase attribute doesn't affect
# the generated ir if ping pong is disabled.
def test_ping_pong(tmpdir):
    ir1 = _get_ir(False, False, False)
    ir2 = _get_ir(True, False, False)
    _compare_irs(ir1, ir2)


# Check that the pipeline stage attribute doesn't affect the
# generated ir if pipelining is disabled.
def test_pipelining(tmpdir):
    ir1 = _get_ir(False, False, False)
    ir2 = _get_ir(False, False, True)
    _compare_irs(ir1, ir2)


# Check that the virtual graph attribute doesn't affect the
# generated ir if virtual graphs are disabled.
def test_virtualgraphs(tmpdir):
    ir1 = _get_ir(False, False, False)
    ir2 = _get_ir(False, True, False)
    _compare_irs(ir1, ir2)
