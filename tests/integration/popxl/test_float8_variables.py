# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import pytest

import numpy as np

import popxl
import popxl.ops as ops


@pytest.mark.parametrize("dtype", [popxl.float8_143, popxl.float8_152])
@pytest.mark.parametrize("use_literal", [True, False])
@pytest.mark.parametrize("log2_scale", [1])
@pytest.mark.parametrize(
    "variable_type",
    [
        "variable",
        "remote_variable",
        "replica_sharded_variable",
        "remote_replica_sharded_variable",
    ],
)
def test_float8_var(dtype, use_literal, log2_scale, variable_type):
    """Test that it is possible to use float8 variables in PopXL."""

    # Some values to load on the device as floats.
    x_shape = [2]
    x_vals = [[1.0, 2.0], [3.0, 4.0]]
    # Some values to load on the device as uint8s.
    x_vals_as_uint8 = [
        popxl.fp8_utils.host_pow2scale_cast_to_fp8(
            x_val, dtype=dtype, log2_scale=log2_scale
        )
        for x_val in x_vals
    ]

    ir = popxl.Ir(replication=2)
    g = ir.main_graph

    with g, popxl.in_sequence():

        if use_literal:
            # Just use normal float values and let the *variable do the conversion.
            x_data = x_vals[0]
            x_args = {"dtype": dtype, "log2_scale": log2_scale, "nan_on_overflow": True}
        else:
            # Pass a value that we converted already.
            x_data = x_vals_as_uint8[0]
            x_args = {"dtype": dtype}

        # Add a normal float8 variable to the graph.
        if variable_type == "variable":
            x = popxl.variable(x_data, **x_args)
            x_on_device = x

        # Add a remote float8 variable to the graph.
        elif variable_type == "remote_variable":
            # Create a buffer for the remote variable.
            buffer = popxl.RemoteBuffer(x_shape, dtype, 1)
            x = popxl.remote_variable(x_data, buffer, 0, **x_args)
            x_on_device = ops.remote_load(buffer, 0)

        # Add a replica sharded float8 variable to the graph.
        elif variable_type == "replica_sharded_variable":
            x, shard_x = popxl.replica_sharded_variable(x_data, **x_args)
            x_on_device = ops.collectives.replicated_all_gather(shard_x)

        elif variable_type == "remote_replica_sharded_variable":
            # Create a buffer for the remote variable.
            buffer = popxl.RemoteBuffer(x_shape, dtype, 1)
            x = popxl.remote_replica_sharded_variable(x_data, buffer, 0, **x_args)
            tmp = ops.remote_load(buffer, 0)
            x_on_device = ops.collectives.replicated_all_gather(tmp)

        # Print the variable to stop it being pruned.
        ops.print_tensor(x_on_device)

    session = popxl.Session(ir, "ipu_model")

    with session:
        # Run the session.
        _ = session.run()

        # Check after a run and retrieving x's value it's the initial value.
        x_val = session.get_tensor_data(x)
        assert np.array_equal(x_val, x_vals_as_uint8[0])

        # Iterate over remaining values.
        for i in range(1, len(x_vals)):
            # Write a new value.
            session.write_variable_data(x, x_vals_as_uint8[i])
            # Run the session.
            _ = session.run()
            # Read it back.
            x_val = session.get_tensor_data(x)
            # Check it matches what we wrote.
            assert np.array_equal(x_val, x_vals_as_uint8[i])
