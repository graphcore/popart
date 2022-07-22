# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import sys
import popxl
import popxl.ops as ops
import numpy as np
from popxl.tensor import Tensor
from pathlib import Path
from typing import Dict
import popart
import popart._internal.ir as _ir

# `import test_util` requires adding to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu


def run_ir(ir: popxl.Ir, bps: int, y_id: str,
           inputs: Dict[str, np.array]) -> np.ndarray:
    """Take the given ir and inputs and run it.

    Args:
        ir (popxl.Ir): The ir to run.
        bps (int): Batches per step.
        y_id (str): The output id, will return this anchor.
        inputs (Dict[str, np.array]): Dict of the inputs to pass to PyStepIO.

    Returns:
        np.array: The anchor associated with y_id.
    """
    _pb_ir = ir._pb_ir  # Internal ir

    dataFlow = popart.DataFlow(
        batchesPerStep=bps,
        anchorTensors={y_id: popart.AnchorReturnType("All")})
    _pb_ir.setDataFlow(dataFlow)

    _pb_ir.updateVertices()

    _pb_ir.setPatterns(
        _ir.patterns.Patterns(_ir.patterns.PatternsLevel.Default))

    with tu.create_test_device() as device:
        session = popart.InferenceSession.fromIr(ir=_pb_ir, deviceInfo=device)

        _pb_ir.logIr()

        session.prepareDevice()

        # Create buffers for anchors
        anchors = session.initAnchorArrays()

        # Run the model
        stepio = popart.PyStepIO(inputs=inputs, outputs=anchors)

        session.weightsFromHost()
        session.run(stepio)

    y = anchors[y_id]
    return y


class AddVariable(popxl.Module):
    def build(self, x, y):
        return x + y


class AddWeights(popxl.Module):
    def __init__(self):
        self.w: Tensor = None

    def build(self, x, y):
        self.w = popxl.graph_input(x.shape, x.dtype, "w")
        return self.w + x + y


class TestConditional:
    def test_variable_inputs(self):
        ir = popxl.Ir()
        opts = ir._pb_ir.getSessionOptions()
        opts.aliasZeroCopy = False
        graph = ir.main_graph
        nx = np.array([23, 10, 22, 21], dtype=np.float32)
        ncond = np.array(True, dtype=bool)
        with graph:
            x = popxl.variable(nx, name="x")
            y = popxl.variable(2 * nx)
            cond = popxl.variable(ncond, name="cond")
            add_weight0 = AddVariable()
            add_weight_graph0 = ir.create_graph(add_weight0, x, y)
            o = ops.conditional(cond=cond,
                                then_branch=add_weight_graph0,
                                else_branch=add_weight_graph0,
                                then_inputs=[x, y],
                                else_inputs=[x, y])[0]
            d2h = popxl.d2h_stream(o.shape,
                                   popxl.dtypes.float,
                                   name="o_stream")
            ops.host_store(d2h, o)
        r_y = run_ir(ir, 1, d2h.tensor_id, {})
        res = 3 * nx
        np.testing.assert_allclose(res, r_y, rtol=1e-8, atol=1e-8)

    def test_slice(self):
        ir = popxl.Ir()
        opts = ir._pb_ir.getSessionOptions()
        opts.aliasZeroCopy = False
        graph = ir.main_graph
        nx = np.array([23, 10, 22, 21], dtype=np.float32)
        nw = np.array([10, 18, 32, 12], dtype=np.float32)
        ncond = np.array(True, dtype=bool)

        with graph:
            x0 = popxl.variable(nx, name="x")
            w = popxl.variable(nw, name="w")
            y = ops.slice(x0, [], [], [])
            cond = popxl.variable(ncond, name="cond")
            add_weight0 = AddWeights()
            add_weight_graph0 = ir.create_graph(add_weight0, x0, x0)
            o = ops.conditional(cond=cond,
                                then_branch=add_weight_graph0,
                                else_branch=add_weight_graph0,
                                then_inputs=[y, x0],
                                else_inputs=[y, x0],
                                then_inputs_dict={add_weight0.w: w},
                                else_inputs_dict={add_weight0.w: w})[0]
            d2h = popxl.d2h_stream(o.shape,
                                   popxl.dtypes.float,
                                   name="o_stream")
            ops.host_store(d2h, o)
        r_y = run_ir(ir, 1, d2h.tensor_id, {})
        res = nw + 2 * nx
        np.testing.assert_allclose(res, r_y, rtol=1e-8, atol=1e-8)
