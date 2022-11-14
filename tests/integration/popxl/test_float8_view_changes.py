# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import popxl
import popxl.ops as ops
from popxl.utils import host_pow2scale_then_cast
import numpy as np
import pytest

data = np.arange(64).reshape(4, 4, 4).astype("float32")


def run_ir(ir: popxl.Ir, y: popxl.Tensor):
    y_d2h = popxl.d2h_stream(y.shape, y.dtype, name="y_stream")
    ops.host_store(y_d2h, y)

    with popxl.Session(ir, "ipu_model") as session:

        outputs = session.run({})

    y_ = outputs[y_d2h]
    return y_


@pytest.mark.parametrize("dtype", [popxl.float8_143, popxl.float8_152])
class TestFloat8ViewChanges:
    """Test various view changes for the special case of float8 tensors.
    All should work as float8 tensors are poplar unsigned char under the hood."""

    @pytest.mark.parametrize("inplace", [True, False])
    def test_float8_slice(self, dtype, inplace):
        """Test you can inplace/outplace slice a float8 tensor."""
        data_in = np.random.random([4, 4, 4])
        data_float8 = host_pow2scale_then_cast(data_in, dtype, 0, True)
        ir = popxl.Ir()
        with ir.main_graph:
            t = popxl.variable(data_float8, dtype=dtype)
            y = None
            if inplace:
                y = ops.slice_(t, start=1, stop=3, step=1, axis=0)
            else:
                y = ops.slice(t, start=1, stop=3, step=1, axis=0)

            assert y.shape == (2, 4, 4)
            assert y.dtype == dtype

            y_host = run_ir(ir, y)

            np.testing.assert_equal(y_host, data_float8[1:3, :, :])

    def test_float8_dynamic_slice(self, dtype):
        """Test you can dynamic slice a float8 tensor."""
        data_in = np.random.random([5, 12, 7])

        axes = [1]
        sizes = [3]
        data_float8 = host_pow2scale_then_cast(data_in, dtype, 0, True)
        ir = popxl.Ir()
        with ir.main_graph:

            t = popxl.variable(data_float8, dtype=dtype)
            index = popxl.variable(np.array([0]))

            y = ops.dynamic_slice(t, index, axes, sizes, True)

            assert y.shape == (5, 3, 7)
            assert y.dtype == dtype

            y_host = run_ir(ir, y)

            np.testing.assert_equal(y_host, data_float8[:, 0:3, :])

    def test_float8_slice_concat(self, dtype):
        """Test you can concat 2 float8 tensors of the same type (even inplace),
        and get back the same sliced tensor."""
        data_in = np.random.random([4, 4, 4])
        data_float8 = host_pow2scale_then_cast(data_in, dtype, 0, True)
        ir = popxl.Ir()
        with ir.main_graph:
            t = popxl.variable(data_float8, dtype=dtype)
            u = popxl.variable(data_float8, dtype=dtype)

            x = ops.slice_(t, start=0, stop=2, step=1, axis=0)
            y = ops.slice_(u, start=2, stop=4, step=1, axis=0)

            assert y.shape == (2, 4, 4)
            assert y.dtype == dtype
            assert x.shape == (2, 4, 4)
            assert x.dtype == dtype

            concat = ops.concat_([x, y], 0)

            assert concat.shape == (4, 4, 4)
            assert concat.dtype == dtype

            y_host = run_ir(ir, concat)

            np.testing.assert_equal(y_host, data_float8)

    def test_float8_slice_concat_different_dtypes(self, dtype):
        """Test that you cannot concat 2 different float8 dtypes"""
        data_in = np.random.random([4, 4, 4])
        data_float8 = host_pow2scale_then_cast(data_in, dtype, 0, True)
        alternate_dtype = (
            popxl.float8_143 if dtype == popxl.float8_152 else popxl.float8_152
        )
        data_float8_alternate = host_pow2scale_then_cast(
            data_in, alternate_dtype, 0, True
        )
        ir = popxl.Ir()
        with ir.main_graph:

            t = popxl.variable(data_float8, dtype=dtype)
            u = popxl.variable(data_float8_alternate)

            x = ops.slice_(t, start=0, stop=2, step=1, axis=0)
            y = ops.slice_(u, start=2, stop=4, step=1, axis=0)

            assert y.shape == (2, 4, 4)
            assert y.dtype == alternate_dtype
            assert x.shape == (2, 4, 4)
            assert x.dtype == dtype
            with pytest.raises(
                TypeError,
                match="All inputs to a concat operation must have the same type:",
            ):
                _ = ops.concat_([x, y], 0)

    @pytest.mark.parametrize("inplace", [True, False])
    def test_float8_reshape(self, dtype, inplace):
        """Test you can inplace/outplace reshape a float8 tensor."""
        data_in = np.random.random([4, 4, 4])
        data_float8 = host_pow2scale_then_cast(data_in, dtype, 0, True)
        ir = popxl.Ir()
        with ir.main_graph:
            t = popxl.variable(data_float8, dtype=dtype)
            y = None
            if inplace:
                y = ops.reshape_(t, (2, 4, 8))
            else:
                y = ops.reshape(t, (2, 4, 8))

            assert y.shape == (2, 4, 8)
            assert y.dtype == dtype

            y_host = run_ir(ir, y)

            np.testing.assert_equal(y_host, np.reshape(data_float8, (2, 4, 8)))

    @pytest.mark.parametrize("inplace", [True, False])
    def test_float8_transpose(self, dtype, inplace):
        """Test you can inplace/outplace transpose a float8 tensor."""
        data_in = np.random.random([1, 2, 3])
        data_float8 = host_pow2scale_then_cast(data_in, dtype, 0, True)
        ir = popxl.Ir()
        with ir.main_graph:
            t = popxl.variable(data_float8, dtype=dtype)
            y = None
            if inplace:
                y = ops.transpose_(t, (0, 2, 1))
            else:
                y = ops.transpose(t, (0, 2, 1))

            assert y.shape == (1, 3, 2)
            assert y.dtype == dtype

            y_host = run_ir(ir, y)

            np.testing.assert_equal(y_host, np.transpose(data_float8, (0, 2, 1)))

    def test_float8_gather(self, dtype):
        """Test you can gather a float8 tensor."""
        data_in = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]).astype(np.float32)
        data_float8 = host_pow2scale_then_cast(data_in, dtype, 0, True)
        indices = np.arange(2, dtype=np.int32)
        ir = popxl.Ir()
        with ir.main_graph:
            t = popxl.variable(data_float8, dtype=dtype)
            yindices = popxl.variable(indices, dtype=popxl.dtypes.int32)

            y = ops.gather(t, yindices)

            assert y.shape == (2,)
            assert y.dtype == dtype

            y_host = run_ir(ir, y)

            np.testing.assert_equal(y_host, np.take(data_float8, indices, 0))

    def test_float8_subsample(self, dtype):
        """Test you can subsample a float8 tensor."""
        shape = [50, 24]
        data_in = np.random.random(shape).astype(np.float32)
        data_float8 = host_pow2scale_then_cast(data_in, dtype, 0, True)
        slices = np.array([2, 4], dtype=np.int32)
        ir = popxl.Ir()
        with ir.main_graph:
            t = popxl.variable(data_float8, dtype=dtype)

            y = ops.subsample(t, slices)

            assert y.shape == (shape[0] // slices[0], shape[1] // slices[1])
            assert y.dtype == dtype

            y_host = run_ir(ir, y)

            np.testing.assert_equal(y_host, data_float8[:: slices[0], :: slices[1]])
