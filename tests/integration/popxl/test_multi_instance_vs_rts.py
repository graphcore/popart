# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import subprocess
import sys

from typing import Optional
from typing_extensions import Literal

import numpy as np
import popxl
import popxl.ops as ops
import popxl.dtypes as dtypes

import pytest


def run_model_and_test():
    import argparse

    parser = argparse.ArgumentParser(description="Parse launch parameters.")
    parser.add_argument("--remote", action="store_true")  # else, on-chip
    parser.add_argument("--with-replica-grouping", action="store_true")
    parser.add_argument("--replica-grouping-stride", type=int, nargs="?", default=1)
    parser.add_argument("--replica-grouping-size", type=int, nargs="?", default=1)
    parser.add_argument("--with-sharding", action="store_true")
    parser.add_argument("--n-shard-over", type=int, nargs="?", default=None)
    parser.add_argument("--retrieval-mode", type=str, nargs="?")
    # argv[:2] is `python run_model_and_test`
    args = parser.parse_args(sys.argv[2:])

    ir = popxl.Ir(replication="popdist")
    mg = ir.main_graph
    """
    Build IR:

    x = [1, 2, ..]
    y = 2
    x += y

    Final x is [3, 4, ...]
    """

    tensor_nelms = 16

    # Initialise:
    #   - w_ir_shape
    #   - replica_grouping
    #   - w_h, v_h buffers with enough data for all groups
    # There is a distinction between when the user passes with_replica_grouping
    # and num_groups=1, vs no replica grouping.
    if args.with_replica_grouping:
        num_groups = ir.replication_factor // args.replica_grouping_size
        w_h = np.ones((num_groups, tensor_nelms), dtype=np.int32)
        # If num_groups is 1, PopXL expects the singleton group dimension to be
        # be squeezed out.
        if num_groups > 1:
            w_ir_shape = w_h.shape[1:]
        else:
            np.squeeze(w_h, axis=0)
            w_ir_shape = w_h.shape
        replica_grouping = ir.replica_grouping(
            stride=args.replica_grouping_stride, group_size=args.replica_grouping_size
        )
    else:
        w_h = np.ones((tensor_nelms,), dtype=np.int32)
        w_ir_shape = w_h.shape
        replica_grouping = None

    w_h = w_h + np.arange(0, w_h.size, dtype=w_h.dtype).reshape(w_h.shape)

    # Init v. Will be broacast in w += v
    v_h = np.ones((1,), dtype=np.int32) * 2

    with mg, popxl.in_sequence():
        if args.remote:
            # Create a remote buffer
            if args.with_sharding:
                buffer = popxl.replica_sharded_buffer(
                    w_h.shape,
                    dtypes.int32,
                    replica_grouping,
                    args.n_shard_over,
                    entries=1,
                )

            else:
                buffer = popxl.remote_buffer(w_ir_shape, dtypes.int32, 1)

            # Create a remote variable whose data comes from the buffer at index 0
            if args.with_sharding:
                remote_w = popxl.remote_replica_sharded_variable(
                    w_h,
                    buffer,
                    0,
                    replica_grouping=replica_grouping,
                    name="remote_w",
                    retrieval_mode=args.retrieval_mode,
                )
            else:
                remote_w = popxl.remote_variable(
                    w_h,
                    buffer,
                    0,
                    replica_grouping=replica_grouping,
                    name="remote_w",
                    retrieval_mode=args.retrieval_mode,
                )

            # Load the remote variable
            w = ops.remote_load(buffer, 0, name="w")
        else:
            w = popxl.variable(
                w_h,
                replica_grouping=replica_grouping,
                name="w",
                retrieval_mode=args.retrieval_mode,
            )

        # Host store `w`. This gets the data this instance's
        # `weights_from_host` sent to the IPU, without relying on
        # `weights_to_host` to send it back correctly. This allows us to test
        # `weights_from_host` in isolation.
        #
        # If sharding, we need to all-gather w (which is just one shard) before
        # host-storing. This is because the poplar tensor for w will have been
        # CBR-rearranged in some unknown way, so host-storing it would result in
        # a tensor that we have no idea how to read.
        #
        # When unsharded, the host-stored tensor from each replica will have the
        # shape of the tensor for the entire group. When sharded, w will have
        # the shape of one shard of the tensor for the entire group. However,
        # after all-gathering, the shape will be for the entire group - the two
        # cases result in host-storing an equivalent tensor.
        if args.with_sharding:
            # Calculate the grouping to be used for the all-gather based on the
            # provided replica grouping and sharding options.
            if args.n_shard_over is None:
                # If no sharding group size specified, we are sharding over the
                # entire group. Thus, the grouping for all-gathering the shards
                # is the same as the replica grouping. If the replica grouping
                # is None, we equivalently pass None here.
                gather_shards_grouping = replica_grouping
            else:
                # If sharding group size is specified, the grouping with which
                # we all-gather the shards has the same stride as the replica
                # grouping, but with the size capped at the sharding group size.
                gather_shards_grouping = ir.replica_grouping(
                    stride=1 if replica_grouping is None else replica_grouping.stride,
                    group_size=args.n_shard_over,
                )

            to_store = ops.collectives.replicated_all_gather(
                w, group=gather_shards_grouping
            )
        else:
            to_store = w

        w_d2h = popxl.d2h_stream(to_store.shape, to_store.dtype)
        ops.host_store(w_d2h, to_store)

        # Perform some calculation on IPU to update the remote-loaded variable.
        v = popxl.constant(v_h, name="v")
        ops.add_(w, v)

        if args.remote:
            # Store the updated value back to the remote buffer
            # This relies on `weights_to_host` to stream back the correct data into
            # the correct places.
            ops.remote_store(buffer, 0, w)

    ir.num_host_transfers = 1
    sess = popxl.Session(ir, "ipu_hw")

    with sess:
        outs = sess.run()

    #######
    # Test weights_from_host:
    # We host-stored the value of `w` immediately after remote_load-ing it. We
    # now compare this value to the initial data w_h. This tests
    # weights_from_host has worked correctly without testing weights_to_host
    # too.
    #
    # Note, if sharding, we would have all-gathered w before host-storing
    # it. This means the shape of the output is (replica, *shape_per_group),
    # not (replica, shape_per_group[0]/num_shards, *shape_per_group[1:]),
    # whether we are sharding or not. That is, every replica is returning the
    # entire data for its group, not just its shard of data for that group.
    #
    # Therefore, the following code to compare the actual data from each replica
    # to the expected data for that replica, will compare the data returned by
    # each replica to the initial w_h data for the entire group that replica
    # belongs to, not just the shard of its group's data that was sent to it by
    # weights_from_host.

    global_replica_offset = ir._pb_ir.getSessionOptions().globalReplicaOffset

    if args.with_replica_grouping:
        assignment = replica_grouping.assignment

    for local_replica in range(ir.instance_replication_factor):
        global_replica = local_replica + global_replica_offset

        # If only 1 group, there is no group dim in w_h, but we still store
        # the group for use in assertion failure messages later.
        if args.with_replica_grouping:
            group = assignment[global_replica]

        # If the data is grouped, get the expected data for this group.
        if args.with_replica_grouping and replica_grouping.num_groups > 1:
            expected = w_h[group]
        else:
            expected = w_h

        # If replicated, get the actual data for this replica.
        if ir.instance_replication_factor > 1:
            actual = outs[w_d2h][local_replica]
        else:
            actual = outs[w_d2h]

        msg = "comparison of host-stored"

        if args.with_sharding:
            if args.n_shard_over is not None:
                num_shards = args.n_shard_over
            elif replica_grouping is not None:
                num_shards = replica_grouping.group_size
            else:
                num_shards = ir.replication_factor
            msg += f", all-gathered (sharded over every {num_shards} replicas of each group)"

        msg += (
            f" initial w for replica {global_replica}, "
            f"which should be equivalent to the expected initial w"
        )

        if args.with_replica_grouping:
            msg += (
                f" for group {group} (num_groups={replica_grouping.num_groups}), "
                f"(group_size={replica_grouping.group_size}, group_stride={replica_grouping.stride})"
            )

        assert_equiv(msg, actual, expected)

    """
    Compare expected updated w_h with actual updated w_h.

    We must compare the correct slices of actual and expected based on the
    global replication factor, replica grouping, and variable retrieval mode:

    one_per_group:
      all cases:
        actual == expected
    all_replicas:
      replicas == 1 (implies ungrouped or groups == 1):
        actual == expected
      replicas > 1, ungrouped or groups == 1:
        actual[replica] == expected
      replicas > 1, grouped and groups > 1:
        actual[replica] == expected[group[replica]]
    """

    expected = w_h + v_h
    actual = sess.get_tensor_data(remote_w if args.remote else w)

    # Passing None would default to one_per_group
    retrieval_mode = (
        args.retrieval_mode if args.retrieval_mode is not None else "one_per_group"
    )

    if retrieval_mode == "one_per_group" or ir.replication_factor == 1:
        assert_equiv("updated w", actual, expected)
    # Must be 'all_replicas'
    # Grouped
    elif args.with_replica_grouping and num_groups > 1:
        # Multi-index expected using the group of each replica
        assert_equiv("updated w", actual, expected[assignment])
    # Ungrouped
    else:
        for replica in range(ir.replication_factor):
            assert_equiv("updated w", actual[replica], expected)


def assert_equiv(desc: str, actual: np.ndarray, expected: np.ndarray):
    assert np.array_equiv(
        actual, expected
    ), f"Equivalence check failed for {desc}:\n  actual =\n{actual}\n  expected =\n{expected}"


def run_multi_instance_test(
    num_instances: int,
    num_global_replicas: int,
    remote: bool = True,
    grouped: bool = False,
    group_size: Optional[int] = None,
    stride: Optional[int] = None,
    sharded: bool = False,
    shard_over: Optional[int] = None,
    retrieval_mode: Optional[
        Literal["one_per_group", "all_replicas"]
    ] = "one_per_group",
):
    """
    Configure and run `run_model_and_test` over multiple instances using poprun.

    `run_model_and_test` runs a simple PopXL program that does an inplace
    addition on a weight.

    The only required parameters are `num_instances` and `num_global_replicas`.

    The other parameters allow us to configure the model. The weight can be
    remote or on-chip, replica-grouped, and/or sharded. You can also choose the
    `retrieval_mode` to use for the variable. Note, remote is the default.

    Generally, this function does not attempt to validate or canonicalise the
    parameters, but instead allow the test author to see what happens when you
    pass the different combinations of parameters to PopXL. Most notably,
    passing grouped=False will result in a `replica_grouping=None` being passed
    when creating the variable in the test; whereas `grouped=True,
    group_size=<num_replicas>` will result in passing a replica_grouping of 1
    group as described. Likewise for the sharding configuration.

    This is why there are separate bool parameters for `grouped` and sharded`,
    instead of just whether or not a grouping/sharding configuration was passed.
    However, for convenience, if you do pass parameters configuring the
    grouping/sharding, we will default grouped/sharded to True for you.

    Note:
        We do not capture the stderr (or stdout) of the subprocess. Instead,
        this function will throw if the poprun subprocess exits with non-zero
        exit code (because of a test failure in one the instances or otherwise).
        You need to scroll up to see the output from the process to see the
        failure that occured; it will not be part of the exception thrown from
        this function.

    Args:
        num_instances (int):
            The number of instances to pass to poprun.
        num_global_replicas (int):
            The global replication factor to pass to poprun.
        remote (bool):
            Whether the variable in the test should be remote or on-chip.
            Defaults to True.
        grouped (bool):
            Whether the variable in the test should have a replica grouping.
            Will be overriden to True if group_size is set, otherwise defaults
            to False.
        group_size (Optional[int], optional):
            The group size to use for the replica grouping. Overrides the
            grouped parameter to True.
        stride (Optional[int], optional):
            The stride to use for the replica grouping. Ignore if grouped=False.
        sharded (bool):
            Whether the variable in the test should be sharded (an RTS variable)
            or not. Overrides remote to True if True. Will be overriden to True
            if shard_over is set, otherwise will default to False.
        shard_over (Optional[int], optional):
            The shard_over to use for the variable (which is the sharding domain
            size). Overrides sharded to True if set.
        retrieval_mode (Optional[Literal["one_per_group", "all_replicas"]]):
            The retrieval mode to use for the variable. Defauts to
            "one_per_group".

    Raises:
        CalledProcessError:
            If the poprun subprocess returns with non-zero exit code.
    """

    """
    How to run this test locally:

    Set the `partition` to any name that is not already taken on your machine.
    PopRun will create/update the partition as appropriate depending on the
    parameters passed to this function.

    `debug` controls how verbose the output is, and whether we output only from
    instance 0 or all instances.
    """
    partition = "popxl--test_multi_instance_vs_rts__"
    debug = False

    # Make it so user does not need to redundantly specify `grouped` if they
    # have specified a `group_size`. Note passing a valid
    # `group_size` is not the concern of this function, we want to see
    # what the test will do when we pass it through.
    if group_size is not None:
        grouped = True

    # Make it so user does not need to redundantly specify `sharded` if they
    # have specified a `shard_over`. Note passing a valid `shard_over` is not
    # the concern of this function, we want to see what the test will do when
    # we pass it through.
    if shard_over is not None:
        sharded = True

    # Default remote to True for the user, if sharded.
    if sharded:
        remote = True

    import pathlib

    test_path = pathlib.Path(__file__).resolve().__str__()

    ipus_per_replica = 1

    command = ["poprun"]

    if debug:
        command.append("-vv")
    else:
        command.append("--only-output-from-instance")
        command.append(str(0))

    command.append("--num-replicas")
    command.append(str(num_global_replicas))
    command.append("--num-instances")
    command.append(str(num_instances))
    command.append("--ipus-per-replica")
    command.append(str(ipus_per_replica))

    command.append("--vipu-partition")
    command.append(partition)
    command.append("--remove-partition")
    command.append("no")
    command.append("--update-partition")
    command.append("yes")

    # So subprocess can find our libraries
    import os

    mpi_local_args = (
        "--mpi-local-args=-x TMPDIR -x LD_LIBRARY_PATH -x PYTHONPATH -x PATH"
    )
    if "POPART_LOG_LEVEL" in os.environ:
        mpi_local_args += " -x POPART_LOG_LEVEL"
    command.append(mpi_local_args)

    py_exe = sys.executable

    command.append(py_exe)
    command.append(test_path)
    command.append("run_model_and_test")

    if remote:
        command.append("--remote")

    if grouped:
        command.append("--with-replica-grouping")
        command.append("--replica-grouping-stride")
        command.append(str(1) if stride is None else str(stride))
        command.append("--replica-grouping-size")
        command.append(str(group_size))

    if sharded:
        command.append("--with-sharding")
        if shard_over:
            command.append("--n-shard-over")
            command.append(str(shard_over))

    if retrieval_mode:
        command.append("--retrieval-mode")
        command.append(retrieval_mode)

    print("Executing command:", command)
    _ = subprocess.run(command, check=True)


@pytest.fixture(params=["one_per_group", "all_replicas"])
def retrieval_mode(request):
    """
    We parameterize a selection of tests over the `retrieval_mode` options.
    There is no particular reasoning for which tests have this, it is just to
    ensure there is good coverage for this option.

    We use this fixture for parameterizing the tests.
    """
    return request.param


def test_no_replica_grouping(retrieval_mode):
    run_multi_instance_test(
        num_instances=2, num_global_replicas=2, retrieval_mode=retrieval_mode
    )


def test_one_group_per_replica():
    run_multi_instance_test(num_instances=2, num_global_replicas=4, group_size=1)


def test_one_group_of_one_replica_per_instance():
    run_multi_instance_test(num_instances=2, num_global_replicas=2, group_size=1)


def test_one_group_of_two_replicas_per_instance(retrieval_mode):
    run_multi_instance_test(
        num_instances=2,
        num_global_replicas=4,
        group_size=2,
        retrieval_mode=retrieval_mode,
    )


def test_orthogonal_groups():
    run_multi_instance_test(
        num_instances=1, num_global_replicas=4, group_size=2, stride=2
    )


def test_orthogonal_groups_within_instances(retrieval_mode):
    run_multi_instance_test(
        num_instances=2,
        num_global_replicas=8,
        group_size=4,
        stride=2,
        retrieval_mode=retrieval_mode,
    )


def test_orthogonal_groups_across_instances():
    run_multi_instance_test(
        num_instances=2, num_global_replicas=8, group_size=2, stride=4
    )


@pytest.mark.parametrize("stride", [2, 4])
def test_multiple_strided_groups_per_instance(stride):
    run_multi_instance_test(
        num_instances=2, num_global_replicas=16, group_size=2, stride=stride
    )


def test_one_group_spanning_all_instances(retrieval_mode):
    run_multi_instance_test(
        num_instances=2,
        num_global_replicas=4,
        group_size=4,
        retrieval_mode=retrieval_mode,
    )


def test_groups_spanning_subset_of_instances():
    run_multi_instance_test(num_instances=4, num_global_replicas=8, group_size=4)


def test_default_sharding():
    run_multi_instance_test(num_instances=1, num_global_replicas=2, sharded=True)


def test_sharding_across_instances_no_local_replication():
    run_multi_instance_test(
        num_instances=2, num_global_replicas=2, group_size=2, sharded=True
    )


def test_sharding_whole_single_instance_groups():
    run_multi_instance_test(
        num_instances=2, num_global_replicas=8, group_size=2, sharded=True
    )


def test_sharding_within_single_instance_groups():
    run_multi_instance_test(
        num_instances=2, num_global_replicas=8, group_size=4, shard_over=2
    )


def test_sharding_whole_cross_instance_groups():
    run_multi_instance_test(
        num_instances=4, num_global_replicas=8, group_size=4, shard_over=4
    )


def test_sharding_within_cross_instance_groups():
    run_multi_instance_test(
        num_instances=4, num_global_replicas=8, group_size=4, shard_over=2
    )


def test_sharding_whole_orthogonal0_cross_instance_groups(retrieval_mode):
    run_multi_instance_test(
        num_instances=4,
        num_global_replicas=8,
        group_size=2,
        stride=4,
        shard_over=2,
        retrieval_mode=retrieval_mode,
    )


@pytest.mark.parametrize("num_instances", [1, 2, 4, 8])
def test_sharding_whole_orthogonal1_cross_instance_groups(num_instances):
    run_multi_instance_test(
        num_instances=num_instances,
        num_global_replicas=8,
        group_size=4,
        stride=2,
        sharded=True,
    )


def test_sharding_within_orthogonal_cross_instance_groups():
    # The replica grouping is not strided, but the intra-group sharding results
    # in a strided collective.
    run_multi_instance_test(
        num_instances=4,
        num_global_replicas=8,
        group_size=4,
        stride=2,
        shard_over=2,
    )


@pytest.mark.parametrize("sharded", [False, True])
def test_sharding_whole_groups_with_stride_that_skips_whole_instances(sharded):
    run_multi_instance_test(
        num_instances=4, num_global_replicas=8, group_size=2, stride=4, sharded=sharded
    )


def test_sharding_within_groups_with_stride_that_skips_whole_instances(retrieval_mode):
    run_multi_instance_test(
        num_instances=16,
        num_global_replicas=16,
        group_size=4,
        stride=4,
        sharded=True,
        shard_over=2,
        retrieval_mode=retrieval_mode,
    )


# On-chip + VariableSettings + multi-instance not supported
@pytest.mark.skip
def test_onchip_group_per_instance():
    run_multi_instance_test(
        num_instances=2, remote=False, num_global_replicas=4, group_size=2
    )


# The first arg passed is the name of the function to execute
if __name__ == "__main__":
    globals()[sys.argv[1]]()
