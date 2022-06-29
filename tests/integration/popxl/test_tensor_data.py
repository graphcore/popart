# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
from typing import Any, Dict, List

import numpy as np
import pytest

import popxl
import popxl.ops as ops
from popxl import dtypes
from popxl.replica_grouping import ReplicaGrouping


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
            _ = ops.add(t, t2)

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
            _ = session.get_tensor_data(b)
        assert "is not of type Constant or Variable" in e_info.value.args[0]

    def test_write_data_constant(self, shape, np_dtype):
        ir = popxl.Ir()
        main = ir.main_graph

        with main:
            arr = get_np_array(shape, np_dtype)
            a = popxl.constant(arr)
            # Add a variable or the graph gets optimized to nothing.
            b = popxl.variable(arr)
            _ = ops.add(a, b)
        ir.num_host_transfers = 1
        session = popxl.Session(ir, device_desc="ipu_model")
        # This should be OK
        _ = session.get_tensor_data(a)

        arr2 = get_np_array(shape, np_dtype)
        # This should throw:
        with session:
            with pytest.raises(TypeError) as e_info:
                session.write_variable_data(a, arr2)
            assert e_info.value.args[0].startswith(
                f"Tensor {a.id} is not of type")

    def test_update_weights(self, shape, np_dtype):
        ir = popxl.Ir()

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
            _ = ops.var_updates.accumulate_(w_tensor, x_tensor)

        ir.num_host_transfers = 1
        session = popxl.Session(ir, device_desc="ipu_model")
        # before update:
        # session.weightsToHost() is not required here. See Tensor.data
        retrieved_array = session.get_tensor_data(w_tensor)
        assert np.allclose(retrieved_array, w_np_data)

        with session:
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
        with session:
            session.write_variable_data(t, input_array_2)
        retrieved_array = session.get_tensor_data(t)

        assert input_array_2.strides == retrieved_array.strides
        assert input_array_2.shape == retrieved_array.shape
        assert np.allclose(input_array_2, retrieved_array)
        assert not np.allclose(input_array, retrieved_array)


def get_np_array(shape, np_dtype):
    return np.array((10 * np.random.random(shape))).astype(np_dtype)


# Some examples of variable settings.
# Adapted from tests/unittests/python/popxl/test_replica_grouping.py
stride_and_size_examples = [
    # ---------------------
    {
        "replicas": 2,
        "stride": 1,
        "group_size": 2,
    },
    # ---------------------
    {
        "replicas": 4,
        "stride": 2,
        "group_size": 2,
    },
    # ----------------------
    {
        "replicas": 4,
        "stride": 1,
        "group_size": 2,
    },
    # ----------------------
    {
        "replicas": 16,
        "stride": 2,
        "group_size": 8,
    },
    # ---------------------- Note: VariableSettings for TMP_4 on a pod256
    {
        "replicas": 256,
        "stride": 64,
        "group_size": 4,
    }
]

CHANNELS = 3
DATA_LEN = 7
O_DIM = 5

input_shape = [O_DIM, CHANNELS, DATA_LEN, DATA_LEN]


@pytest.mark.parametrize("settings", stride_and_size_examples)
@pytest.mark.parametrize("retrieval_mode", ["one_per_group", "all_replicas"])
class TestReplicaGroupedVariables:
    def _get_weights_array(self, shape: List[int],
                           replica_grouping: ReplicaGrouping) -> np.ndarray:
        """Get a correctly shaped numpy array given the provided arguments.

        Args:
            shape (List[int]): The tensor shape on device.
            replica_grouping (ReplicaGrouping) The replica grouping used to create
                the variable. Optional.

        Returns:
            np.array: An np.array of the correct size for use in this case.
        """
        reshape = []
        if replica_grouping.num_groups > 1:
            reshape = [replica_grouping.num_groups]
        reshape.extend(shape)

        array = np.random.random_sample(reshape).astype(np.float32)
        return array

    def test_get_and_set_replica_grouped_variables(
            self, settings: Dict[str, Any], retrieval_mode: str) -> None:
        """Test the getting and setting of variables works with replica grouping.

        Args:
            settings (Dict[str, Any]): A samples of replica grouping settings to use.
        """
        repeat = lambda x, n: np.repeat(x[None, ...], n, axis=0)

        ir = popxl.Ir()
        ir.replication_factor = settings["replicas"]
        main = ir.main_graph

        with main:
            replica_grouping = ir.replica_grouping(settings["stride"],
                                                   settings["group_size"])
            data = self._get_weights_array(input_shape, replica_grouping)

            # Get this data reversed to write and check it has updated.
            data_reversed = np.ascontiguousarray(np.flip(data))

            v1 = popxl.variable(data,
                                name="v1",
                                replica_grouping=replica_grouping,
                                retrieval_mode=retrieval_mode)

            d2h_stream = popxl.d2h_stream(v1.shape, v1.dtype, 'v1_stream')
            ops.host_store(d2h_stream, v1)

        session = popxl.Session(ir, device_desc='ipu_model')
        assignment = np.array(replica_grouping.assignment)

        with session:
            if retrieval_mode == 'one_per_group':
                session.write_variable_data(v1, data_reversed)
            else:
                with pytest.raises(NotImplementedError):
                    session.write_variable_data(v1, data_reversed)
                return

            v1_retrieved = session.get_tensor_data(v1)

            if retrieval_mode == 'one_per_group':
                np.testing.assert_allclose(v1_retrieved, data_reversed)
            else:
                if replica_grouping.num_groups == 1:
                    data2 = repeat(data_reversed, settings["replicas"])
                    np.testing.assert_allclose(v1_retrieved, data2)
                else:
                    data2 = np.take(data_reversed, assignment, axis=0)
                    np.testing.assert_allclose(v1_retrieved, data2)

            output = session.run()
            t = output[d2h_stream]

            assert len(t) == settings["replicas"]

            if replica_grouping.num_groups == 1:
                np.testing.assert_allclose(
                    t, repeat(data_reversed, settings["replicas"]))
            else:
                for group in range(0, replica_grouping.num_groups):
                    group_mask = assignment == group
                    np.testing.assert_allclose(
                        t[group_mask],
                        repeat(data_reversed[group], settings["group_size"]))
