# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import itertools

import numpy as np
import pytest

import popxl
from popxl import ops

NUM_REPLICAS = 4
DTYPE = np.int32


def add_replicated_all_gather(x, group):
    return ops.collectives.replicated_all_gather(x, group)


def add_replicated_all_reduce(x, group):
    return ops.collectives.replicated_all_reduce(x, "add", group)


def add_replicated_all_reduce_(x, group):
    return ops.collectives.replicated_all_reduce_(x, "add", group)


def add_replicated_reduce_scatter(x, group):
    return ops.collectives.replicated_reduce_scatter(x, "add", group)


def get_replicated_all_gather_expected_result(group):
    """Get expected data for the test cases that use a replicated all gather in
    `test_output_is_numerically_correct`.

    In these tests each replica has a variable `x` which contains a single integer equal
    to the index of its replica.

    When a replicated all gather is performed, each replica will now have a new tensor
    `y` which contains the values of `x` from the replicas in its replica grouping.
    Because each `x` has been initialised with a value equal to its replica index, `y`
    will contain the indices of all replicas in its replica grouping.

    For example consider the following initialisation of `x` with 4 replicas, where the
    number after `x` reflects the replica index:

    ```
    x_0 = [0]
    x_1 = [1]
    x_2 = [2]
    x_3 = [3]
    ```

    Consider a replica grouping with a stride of 2 and group size of 2. This means that
    replicas 0 and 2 are in a group and 1 and 3 are in another group.

    The resulting `y`s on each replica after a replicated all gather will be:

    ```
    y_0 = [0, 2]
    y_1 = [1, 3]
    y_2 = [0, 2]
    y_3 = [1, 3]
    ```
    """
    result = []
    assignment = group.assignment
    for i in range(NUM_REPLICAS):
        tmp = np.array(assignment) == assignment[i]
        tmp = np.argwhere(tmp)
        result.append(tmp)
    return np.array(result, dtype=DTYPE)


def get_replicated_all_reduce_expected_result(group):
    """Get expected data for the test cases that use a replicated all reduce in
    `test_output_is_numerically_correct`.

    Since the test use the 'add' reduction method, the result would be the same as if we
    did a replicated all gather and then summed all values in the gathered tensor.
    """
    result = get_replicated_all_gather_expected_result(group)[:, :, 0].sum(axis=1)
    return result[:, None]


def get_replicated_all_reduce__expected_result(group):
    """Get expected data for the test cases that use an inplace replicated all reduce in
    `test_output_is_numerically_correct`. The output is the same as the out-of-place
    case.
    """
    return get_replicated_all_reduce_expected_result(group)


def get_replicated_reduce_scatter_expected_result(group):
    """Get expected data for the test cases that use a replicated reduce scatter in
    `test_output_is_numerically_correct`.

    After the replicated reduce scatter, the first replica in each group will have the
    sum of values of all replicas in its group. All other replicas will have a value of
    0.
    """
    result = []
    assignment = group.assignment
    for i in range(NUM_REPLICAS):
        tmp = np.array(assignment) == assignment[i]
        tmp = np.argwhere(tmp)
        if i == tmp[0]:
            result.append([tmp.sum()])
        else:
            result.append([0])
    return np.array(result, dtype=DTYPE)


def get_test_output_is_numerically_correct_data():
    """Generate test data for `test_output_is_numerically_correct` in a single session
    run.
    """
    ir = popxl.Ir(NUM_REPLICAS)

    groupings = [
        ir.replica_grouping(stride, group_size)
        for stride, group_size in ((1, 1), (1, 2), (2, 2), (1, 4))
    ]
    collective_names = [
        "replicated_all_gather",
        "replicated_all_reduce",
        "replicated_all_reduce_",
        "replicated_reduce_scatter",
    ]
    test_cases = lambda: itertools.product(collective_names, groupings)
    streams = []

    with ir.main_graph, popxl.in_sequence():
        for collective_name, grouping in test_cases():
            # We start by adding a variable with a single element. This variable will be
            # initialised with an integer equal to the replica index it belongs to.
            data = np.arange(NUM_REPLICAS, dtype=DTYPE)[:, None]
            x = popxl.variable(data, replica_grouping=ir.replica_grouping(1, 1))
            # This is one of `add_replicated_all_gather`, `add_replicated_all_reduce`,
            # `add_replicated_all_reduce_` from above.
            add_collective = eval(f"add_{collective_name}")
            y = add_collective(x, grouping)
            stream = popxl.d2h_stream(y.shape, y.dtype)
            ops.host_store(stream, y)
            streams.append(stream)

    with popxl.Session(ir, "ipu_hw") as session:
        outputs = session.run()

    for stream, (collective_name, grouping) in zip(streams, test_cases()):
        yield collective_name, grouping, outputs[stream]


@pytest.mark.parametrize(
    "collective_name,grouping,model_output",
    get_test_output_is_numerically_correct_data(),
)
def test_output_is_numerically_correct(collective_name, grouping, model_output):
    """Test that the output of a collective operation is equal to a reference output.

    To avoid compiling an executable and acquiring a device for each executable, we call
    `get_test_output_is_numerically_correct_data` once to generate the model outputs in
    a single run. This test simply compares the outputs to the reference ones.
    """
    get_expected_result = eval(f"get_{collective_name}_expected_result")
    expected_result = get_expected_result(grouping)
    np.testing.assert_array_equal(model_output, expected_result)
