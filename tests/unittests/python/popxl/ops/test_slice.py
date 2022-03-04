# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popxl
import popxl.ops as ops
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


def run_ir(ir: popxl.Ir, y: popxl.Tensor):
    ir_ = ir._pb_ir  # Internal ir
    y_d2h = popxl.d2h_stream(y.shape, y.dtype, name="y_stream")
    ops.host_store(y_d2h, y)
    y_id = y_d2h.tensor_id

    dataFlow = popart.DataFlow(
        batchesPerStep=1, anchorTensors={y_id: popart.AnchorReturnType("All")})
    ir_.setDataFlow(dataFlow)

    opts = ir_.getSessionOptions()
    opts.useHostCopyOps = True
    opts.enableExplicitMainLoops = True
    opts.aliasZeroCopy = True
    opts.explicitRecomputation = True

    ir_.updateVertices()

    with tu.create_test_device() as device:
        session = popart.InferenceSession.fromIr(ir=ir_, deviceInfo=device)

        session.prepareDevice()

        # Create buffers for anchors
        anchors = session.initAnchorArrays()

        # Run the model
        stepio = popart.PyStepIO(inputs={}, outputs=anchors)

        session.weightsFromHost()
        session.run(stepio)

    y_ = anchors[y_id]
    return y_


class TestSlice:
    @pytest.mark.parametrize("inplace", [True, False])
    def test_fn(self, inplace):
        ir = popxl.Ir()
        g = ir.main_graph
        with g:
            t = popxl.variable(data)
            if inplace:
                y = ops.slice_(t, start=1, stop=3, step=1, axis=0)
            else:
                y = ops.slice(t, start=1, stop=3, step=1, axis=0)

        if not inplace:
            assert contains_op_of_type("Slice", _ir.op.SliceOp, g)
        else:
            assert contains_op_of_type("SliceInplace", _ir.op.SliceInplaceOp,
                                       g)
        assert len(g.tensors) == 2
        assert len(g.variables) == 1

    @pytest.mark.parametrize("inplace", [True, False])
    def test_fn_numerically(self, inplace):
        ir = popxl.Ir()
        g = ir.main_graph
        with g:
            t = popxl.variable(data)
            if inplace:
                y = ops.slice_(t, start=1, stop=3, step=1, axis=0)
            else:
                y = ops.slice(t, start=1, stop=3, step=1, axis=0)
            y_host = run_ir(ir, y)

        y_numpy = data[1:3]
        assert_array_equal(y_host, y_numpy)

    @pytest.mark.parametrize("inplace", [True, False])
    def test_start_only(self, inplace):
        ir = popxl.Ir()
        with ir.main_graph:
            t = popxl.variable(data)
            if inplace:
                y = ops.slice_(t, start=1)
            else:
                y = ops.slice(t, start=1)
            y_host = run_ir(ir, y)

        y_numpy = data[1:]
        assert_array_equal(y_host, y_numpy)

    @pytest.mark.parametrize("inplace", [True, False])
    def test_start_only_multidim(self, inplace):
        ir = popxl.Ir()
        with ir.main_graph:
            t = popxl.variable(data)
            if inplace:
                y = ops.slice_(t, start=[1, 2])
            else:
                y = ops.slice(t, start=[1, 2])
            y_host = run_ir(ir, y)

        y_numpy = data[1:, 2:]
        assert_array_equal(y_host, y_numpy)

    @pytest.mark.parametrize("inplace", [True, False])
    def test_stop_only(self, inplace):
        ir = popxl.Ir()
        with ir.main_graph:
            t = popxl.variable(data)
            if inplace:
                y = ops.slice_(t, stop=2)
            else:
                y = ops.slice(t, stop=2)
            y_host = run_ir(ir, y)

        y_numpy = data[:2]
        assert_array_equal(y_host, y_numpy)

    @pytest.mark.parametrize("inplace", [True, False])
    def test_stop_only_multidim(self, inplace):
        ir = popxl.Ir()
        with ir.main_graph:
            t = popxl.variable(data)
            if inplace:
                y = ops.slice_(t, stop=[2, 3])
            else:
                y = ops.slice(t, stop=[2, 3])
            y_host = run_ir(ir, y)

        y_numpy = data[:2, :3]
        assert_array_equal(y_host, y_numpy)

    @pytest.mark.parametrize("inplace", [True, False])
    def test_identity_fn(self, inplace):
        ir = popxl.Ir()
        with ir.main_graph:
            t = popxl.variable(data)
            if inplace:
                y = ops.slice_(t, axis=0)  # `axis=0` is redundant
            else:
                y = ops.slice(t, axis=0)  # `axis=0` is redundant

        assert len(ir.main_graph.tensors) == 1
        assert len(ir.main_graph.variables) == 1

    @pytest.mark.parametrize("inplace", [True, False])
    def test_identity_numerically(self, inplace):
        ir = popxl.Ir()
        with ir.main_graph:
            t = popxl.variable(data)
            if inplace:
                y = ops.slice_(t, axis=0)  # `axis=0` is redundant
            else:
                y = ops.slice(t, axis=0)  # `axis=0` is redundant
            y_host = run_ir(ir, y)

        assert_array_equal(y_host, data)

    @pytest.mark.parametrize("inplace", [True, False])
    def test_start_and_stop(self, inplace):
        ir = popxl.Ir()
        with ir.main_graph:
            t = popxl.variable(data)
            if inplace:
                y = ops.slice_(t, start=[1, 2], stop=[3, 4])
            else:
                y = ops.slice(t, start=[1, 2], stop=[3, 4])
            y_host = run_ir(ir, y)

        y_numpy = data[1:3, 2:4]
        assert_array_equal(y_host, y_numpy)

    @pytest.mark.parametrize("inplace", [True, False])
    def test_step(self, inplace):
        ir = popxl.Ir()
        with ir.main_graph:
            t = popxl.variable(data)
            if inplace:
                y = ops.slice_(t, start=[1, 3], stop=[3, 1], step=[1, -1])
            else:
                y = ops.slice(t, start=[1, 3], stop=[3, 1], step=[1, -1])
            y_host = run_ir(ir, y)

        y_numpy = data[1:3, 3:1:-1]
        assert_array_equal(y_host, y_numpy)

    @pytest.mark.parametrize("inplace", [True, False])
    def test_negative_start(self, inplace):
        ir = popxl.Ir()
        with ir.main_graph:
            t = popxl.variable(data)
            if inplace:
                y = ops.slice_(t, start=-2, step=-1)
            else:
                y = ops.slice(t, start=-2, step=-1)
            y_host = run_ir(ir, y)

        y_numpy = data[-2::-1]
        assert_array_equal(y_host, y_numpy)

    @pytest.mark.parametrize("inplace", [True, False])
    def test_axis(self, inplace):
        ir = popxl.Ir()
        with ir.main_graph:
            t = popxl.variable(data)
            if inplace:
                y = ops.slice_(t, start=[1, 2], stop=[3, 4], axis=[2, 1])
            else:
                y = ops.slice(t, start=[1, 2], stop=[3, 4], axis=[2, 1])
            y_host = run_ir(ir, y)

        y_numpy = data[:, 2:4, 1:3]
        assert_array_equal(y_host, y_numpy)

    @pytest.mark.parametrize("inplace", [True, False])
    def test_error_lengths(self, inplace):
        ir = popxl.Ir()
        with ir.main_graph:
            t = popxl.variable(data)
            with pytest.raises(ValueError):
                if inplace:
                    y = ops.slice_(t, start=[2], stop=[3, 4], axis=[2, 1])
                else:
                    y = ops.slice(t, start=[2], stop=[3, 4], axis=[2, 1])

    def test_dunder_scalar(self):
        ir = popxl.Ir()
        with ir.main_graph:
            t = popxl.variable(data)
            y = t[0]
            y_host = run_ir(ir, y)

        y_numpy = data[0]
        assert_array_equal(y_host, y_numpy)

    def test_dunder_slice(self):
        ir = popxl.Ir()
        with ir.main_graph:
            t = popxl.variable(data)
            y = t[0:2]
            y_host = run_ir(ir, y)

        y_numpy = data[0:2]
        assert_array_equal(y_host, y_numpy)

    def test_dunder_scalar_and_slice(self):
        ir = popxl.Ir()
        with ir.main_graph:
            t = popxl.variable(data)
            y = t[0, 3:0:-1, 2]
            y_host = run_ir(ir, y)

        y_numpy = data[0, 3:0:-1, 2]
        assert_array_equal(y_host, y_numpy)
