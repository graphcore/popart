# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart.ir as pir
import popart.ir.ops as ops
import popart._internal.ir as _ir
import popart
from utils import contains_op_of_type
import numpy as np
from numpy.testing import assert_array_equal
import pytest

# `import test_util` requires adding to sys.path
import sys
from pathlib import Path
sys.path.append(
    str(
        Path(__file__).resolve().parent.parent.parent.parent.parent /
        "integration"))
import test_util as tu

data = np.arange(64).reshape(4, 4, 4).astype('float32')


def run_ir(ir, y):
    ir = ir._pb_ir  # Internal ir
    y_d2h = pir.d2h_stream(y.shape, y.dtype, name="y_stream")
    ops.host_store(y_d2h, y)
    y_id = y_d2h.tensor_id()

    dataFlow = popart.DataFlow(
        batchesPerStep=1, anchorTensors={y_id: popart.AnchorReturnType("All")})
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

    y = anchors[y_id]
    return y


class TestSlice:
    def test_fn(self):
        ir = pir.Ir()
        g = ir.main_graph()
        with g:
            t = pir.variable(data)
            y = ops.slice(t, start=1, stop=3, step=1, axis=0)

        assert contains_op_of_type("Slice", _ir.op.SliceOp, g)
        assert len(g.get_tensors()) == 2
        assert len(g.get_variables()) == 1

    def test_fn_numerically(self):
        ir = pir.Ir()
        g = ir.main_graph()
        with g:
            t = pir.variable(data)
            y = ops.slice(t, start=1, stop=3, step=1, axis=0)
            y_host = run_ir(ir, y)

        y_numpy = data[1:3]
        assert_array_equal(y_host, y_numpy)

    def test_start_only(self):
        ir = pir.Ir()
        with ir.main_graph():
            t = pir.variable(data)
            y = ops.slice(t, start=1)
            y_host = run_ir(ir, y)

        y_numpy = data[1:]
        assert_array_equal(y_host, y_numpy)

    def test_start_only_multidim(self):
        ir = pir.Ir()
        with ir.main_graph():
            t = pir.variable(data)
            y = ops.slice(t, start=[1, 2])
            y_host = run_ir(ir, y)

        y_numpy = data[1:, 2:]
        assert_array_equal(y_host, y_numpy)

    def test_stop_only(self):
        ir = pir.Ir()
        with ir.main_graph():
            t = pir.variable(data)
            y = ops.slice(t, stop=2)
            y_host = run_ir(ir, y)

        y_numpy = data[:2]
        assert_array_equal(y_host, y_numpy)

    def test_stop_only_multidim(self):
        ir = pir.Ir()
        with ir.main_graph():
            t = pir.variable(data)
            y = ops.slice(t, stop=[2, 3])
            y_host = run_ir(ir, y)

        y_numpy = data[:2, :3]
        assert_array_equal(y_host, y_numpy)

    def test_identity_fn(self):
        ir = pir.Ir()
        with ir.main_graph():
            t = pir.variable(data)
            y = ops.slice(t, axis=0)  # `axis=0` is redundant

        assert len(ir.main_graph().get_tensors()) == 1
        assert len(ir.main_graph().get_variables()) == 1

    def test_identity_numerically(self):
        ir = pir.Ir()
        with ir.main_graph():
            t = pir.variable(data)
            y = ops.slice(t, axis=0)  # `axis=0` is redundant
            y_host = run_ir(ir, y)

        assert_array_equal(y_host, data)

    def test_start_and_stop(self):
        ir = pir.Ir()
        with ir.main_graph():
            t = pir.variable(data)
            y = ops.slice(t, start=[1, 2], stop=[3, 4])
            y_host = run_ir(ir, y)

        y_numpy = data[1:3, 2:4]
        assert_array_equal(y_host, y_numpy)

    def test_step(self):
        ir = pir.Ir()
        with ir.main_graph():
            t = pir.variable(data)
            y = ops.slice(t, start=[1, 3], stop=[3, 1], step=[1, -1])
            y_host = run_ir(ir, y)

        y_numpy = data[1:3, 3:1:-1]
        assert_array_equal(y_host, y_numpy)

    def test_negative_start(self):
        ir = pir.Ir()
        with ir.main_graph():
            t = pir.variable(data)
            y = ops.slice(t, start=-2, step=-1)
            y_host = run_ir(ir, y)

        y_numpy = data[-2::-1]
        assert_array_equal(y_host, y_numpy)

    def test_axis(self):
        ir = pir.Ir()
        with ir.main_graph():
            t = pir.variable(data)
            y = ops.slice(t, start=[1, 2], stop=[3, 4], axis=[2, 1])
            y_host = run_ir(ir, y)

        y_numpy = data[:, 2:4, 1:3]
        assert_array_equal(y_host, y_numpy)

    def test_error_lengths(self):
        ir = pir.Ir()
        with ir.main_graph():
            t = pir.variable(data)
            with pytest.raises(ValueError):
                y = ops.slice(t, start=[2], stop=[3, 4], axis=[2, 1])

    def test_dunder_scalar(self):
        ir = pir.Ir()
        with ir.main_graph():
            t = pir.variable(data)
            y = t[
                0]  # Axis isn't reduced automatically. See docstring of `__getitem__`
            y_host = run_ir(ir, y)

        y_numpy = data[0:1]
        assert_array_equal(y_host, y_numpy)

    def test_dunder_slice(self):
        ir = pir.Ir()
        with ir.main_graph():
            t = pir.variable(data)
            y = t[0:2]
            y_host = run_ir(ir, y)

        y_numpy = data[0:2]
        assert_array_equal(y_host, y_numpy)

    def test_dunder_scalar_and_slice(self):
        ir = pir.Ir()
        with ir.main_graph():
            t = pir.variable(data)
            y = t[
                0, 3:0:
                -1]  # Axis isn't reduced automatically. See docstring of `__getitem__`
            y_host = run_ir(ir, y)

        y_numpy = data[0:1, 3:0:-1]
        assert_array_equal(y_host, y_numpy)
