# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart
import popart.ir as pir
import popart.ir.ops as ops

# `import test_util` requires adding to sys.path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu


def test_modified():
    ir = pir.Ir()
    g = ir.main_graph()

    with g, pir.in_sequence():
        x = pir.variable(1)

        sg = ir.create_graph(
            lambda x: ops.var_updates.accumulate_(x, pir.constant(1)), x)

        ops.call(sg, x)  # type: ignore
        # Store x
        x_non_modify_stream = pir.d2h_stream(x.shape, x.dtype)
        ops.host_store(x_non_modify_stream, x)

        info = ops.call_with_info(sg, x)
        info.set_op_input_modified(x)
        x_modifiy_stream = pir.d2h_stream(x.shape, x.dtype)
        ops.host_store(x_modifiy_stream, x)

    ir = ir._pb_ir
    dataFlow = popart.DataFlow(batchesPerStep=1,
                               anchorTensors={
                                   x_non_modify_stream.tensor_id():
                                   popart.AnchorReturnType("All"),
                                   x_modifiy_stream.tensor_id():
                                   popart.AnchorReturnType("All"),
                               })
    ir.setDataFlow(dataFlow)

    opts = ir.getSessionOptions()
    opts.useHostCopyOps = True
    opts.enableExplicitMainLoops = True
    opts.aliasZeroCopy = True
    opts.explicitRecomputation = True

    ir.updateVertices()
    ir.setIsPrepared()

    session = popart.InferenceSession.fromIr(
        ir=ir, deviceInfo=tu.create_test_device())

    session.prepareDevice()

    # Create buffers for anchors
    anchors = session.initAnchorArrays()

    # Run the model
    stepio = popart.PyStepIO(inputs={}, outputs=anchors)

    session.weightsFromHost()
    session.run(stepio)

    assert anchors[x_non_modify_stream.tensor_id()] == 1
    assert anchors[x_modifiy_stream.tensor_id()] == 2
