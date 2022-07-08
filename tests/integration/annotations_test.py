# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import popart
import numpy as np
import json
import test_util as tu


def _get_ir(executionphases_enabled, virtualgraph_enabled, pipeline_enabled):
    dsize = 10
    builder = popart.Builder()
    ip = builder.addInputTensor(popart.TensorInfo("FLOAT", [dsize, dsize]))

    anchorIds = []
    anchorIds.append(popart.reservedGradientPrefix() + ip)

    x = ip
    for i in range(3):
        w = builder.addInitializedInputTensor(np.ones([dsize, dsize], np.float32))
        x = builder.aiOnnx.matmul([x, w])

        if executionphases_enabled:
            builder.executionPhase(x, i)
        if virtualgraph_enabled:
            builder.virtualGraph(x, i)
        if pipeline_enabled:
            builder.pipelineStage(x, i)

        anchorIds.append(popart.reservedGradientPrefix() + x)

    out = builder.aiGraphcore.identityloss([x])
    if virtualgraph_enabled:
        builder.virtualGraph(out, 3)

    with tu.create_test_device() as device:
        dfAnchors = {}
        for anchorId in anchorIds:
            dfAnchors.update({anchorId: popart.AnchorReturnType("All")})

        opts = popart.SessionOptions()
        # disable outlining to make the ir easier to parse
        opts.enableOutlining = False

        proto = builder.getModelProto()

        session = popart.TrainingSession(
            fnModel=proto,
            dataFlow=popart.DataFlow(1, dfAnchors),
            optimizer=popart.ConstSGD(0.1),
            loss=out,
            patterns=popart.Patterns(popart.PatternsLevel.All),
            userOptions=opts,
            deviceInfo=device,
        )

        ir = json.loads(session._serializeIr(popart.IrSerializationFormat.JSON))
    return ir


def _compare_irs(ir1, ir2):
    # there should only be one graph in the ir
    assert len(ir1) == 1
    assert len(ir2) == 1

    g1 = ir1["maingraph"]
    g2 = ir2["maingraph"]

    # graphs should have the same number of ops
    assert len(g1) == len(g2)

    def get_op_comparison_string(op):
        name = op["name"]
        op_type = op["type"]
        if name:
            return f"{name}:{op_type}"
        else:
            return f"{op_type}"

    for op1, op2 in zip(g1, g2):
        op1 = get_op_comparison_string(op1)
        op2 = get_op_comparison_string(op2)
        print(f"{op1:<60} | {op2:<60} | {op1 == op2}")
        assert op1 == op2


# Check that the execution phase attribute doesn't affect
# the generated ir if phased execution is disabled.
def test_ping_pong():
    ir1 = _get_ir(False, False, False)
    ir2 = _get_ir(True, False, False)
    _compare_irs(ir1, ir2)


# Check that the pipeline stage attribute doesn't affect the
# generated ir if pipelining is disabled.
def test_pipelining():
    ir1 = _get_ir(False, False, False)
    ir2 = _get_ir(False, False, True)
    _compare_irs(ir1, ir2)


# Check that the virtual graph attribute doesn't affect the
# generated ir if virtual graphs are disabled.
def test_virtualgraphs():
    ir1 = _get_ir(False, False, False)
    ir2 = _get_ir(False, True, False)
    _compare_irs(ir1, ir2)
