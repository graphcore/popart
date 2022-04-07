# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import numpy as np
import pytest

import popxl
import popxl.ops as ops
from popxl import dtypes


@pytest.mark.parametrize("shape", [[2, 4, 1, 3], []])
@pytest.mark.parametrize("np_dtype",
                         [np.float32, np.float16, np.int32, np.uint32])
class TestData:
    def test_data_variable(self, shape, np_dtype):
        ir = popxl.Ir()
        main = ir.main_graph

        with main:
            input_array = get_np_array(shape, np_dtype)
            t = popxl.variable(input_array)
            _ = ops.add(t, t)

        ir.num_host_transfers = 1
        session = popxl.Session(ir, device_desc="ipu_model")
        retrieved_array = session.get_tensor_data(t)

        assert input_array.strides == retrieved_array.strides
        assert input_array.shape == retrieved_array.shape
        assert np.allclose(input_array, retrieved_array)

    def test_data_constant(self, shape, np_dtype):
        ir = popxl.Ir()
        main = ir.main_graph

        with main:
            arr = get_np_array(shape, np_dtype)
            t = popxl.constant(arr)
            t2 = popxl.variable(arr)
            a = ops.add(t, t2)

        ir.num_host_transfers = 1
        session = popxl.Session(ir, device_desc="ipu_model")
        retrieved_array = session.get_tensor_data(t)

        assert arr.strides == t.strides()
        assert arr.shape == retrieved_array.shape
        assert np.allclose(arr, retrieved_array)

    def test_data_actgrad(self, shape, np_dtype):
        ir = popxl.Ir()
        main = ir.main_graph

        with main:
            arr = get_np_array(shape, np_dtype)
            a = popxl.variable(arr)
            b = ops.add(a, a)
        ir.num_host_transfers = 1
        session = popxl.Session(ir, device_desc="ipu_model")
        with pytest.raises(TypeError) as e_info:
            retrieved_array = session.get_tensor_data(b)
        assert e_info.value.args[0].startswith(f"Tensor {b.id} is not of type")

    def test_write_data_constant(self, shape, np_dtype):
        ir = popxl.Ir()
        main = ir.main_graph

        with main:
            arr = get_np_array(shape, np_dtype)
            a = popxl.constant(arr)
            # Add a variable or the graph gets optimized to nothing.
            b = popxl.variable(arr)
            c = ops.add(a, b)
        ir.num_host_transfers = 1
        session = popxl.Session(ir, device_desc="ipu_model")
        # This should be OK
        retrieved_array = session.get_tensor_data(a)

        arr2 = get_np_array(shape, np_dtype)
        # This should throw:

        with pytest.raises(TypeError) as e_info:
            session.write_variable_data(a, arr2)
        assert e_info.value.args[0].startswith(f"Tensor {a.id} is not of type")

    def test_update_weights(self, shape, np_dtype):
        ir = popxl.Ir()
        ir_ = ir._pb_ir  # Internal ir

        g = ir.main_graph
        w_np_data = get_np_array(shape, np_dtype)
        w_np_data_copy = w_np_data.copy(
        )  # make sure the numpy array isn't updated.
        x_data = get_np_array(shape, np_dtype)
        with g:
            x_d2h = popxl.h2d_stream(x_data.shape,
                                     dtypes.dtype.as_dtype(x_data.dtype),
                                     name="x_stream")
            x_tensor = ops.host_load(x_d2h, "x")
            w_tensor = popxl.variable(w_np_data, name="weight_data")
            updated_w = ops.var_updates.accumulate_(w_tensor, x_tensor)

        ir.num_host_transfers = 1
        session = popxl.Session(ir, device_desc="ipu_model")
        # before update:
        # session.weightsToHost() is not required here. See Tensor.data
        retrieved_array = session.get_tensor_data(w_tensor)
        assert np.allclose(retrieved_array, w_np_data)

        _ = session.run({x_d2h: x_data})

        print("w_data: ", retrieved_array)
        print("expected: ", w_np_data_copy + x_data)
        # Weird stuff happens at float16
        atol = 1e-08 if not np_dtype == np.float16 else 1e-04
        rtol = 1e-05 if not np_dtype == np.float16 else 1e-03
        # After update:
        assert np.allclose(session.get_tensor_data(w_tensor),
                           w_np_data_copy + x_data, rtol,
                           atol)  #  <-- should be updated.


@pytest.mark.parametrize("shape", [[2, 4, 1, 3], []])
@pytest.mark.parametrize("np_dtype",
                         [np.float32, np.float16, np.int32, np.uint32])
class TestWriteTensorData:
    def test_write_data_variable(self, shape, np_dtype):
        ir = popxl.Ir()
        main = ir.main_graph

        with main:
            input_array = get_np_array(shape, np_dtype)
            t = popxl.variable(input_array)
            _ = ops.add(t, t)

        ir.num_host_transfers = 1
        session = popxl.Session(ir, device_desc="ipu_model")
        input_array_2 = get_np_array(shape, np_dtype)
        session.write_variable_data(t, input_array_2)
        retrieved_array = session.get_tensor_data(t)

        assert input_array_2.strides == retrieved_array.strides
        assert input_array_2.shape == retrieved_array.shape
        assert np.allclose(input_array_2, retrieved_array)
        assert not np.allclose(input_array, retrieved_array)


def get_np_array(shape, np_dtype):
    return np.array((10 * np.random.random(shape))).astype(np_dtype)
