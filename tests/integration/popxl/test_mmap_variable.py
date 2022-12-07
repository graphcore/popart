# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import numpy as np
import popxl
import popxl.ops
import pytest
import pathlib
from types import MethodType
from typing import Tuple, Generator, Callable, Optional

from popxl_test_device_helpers import mk_session_with_test_device

"""
The following functions are fixtures that the test is parameterised on.
They abstract over the possible Variable creation APIs in PopXL.
Their setup will create a Variable and, if appropriate, remote_load a Tensor.
Their teardown will remote_store the tensor.
"""
_TestFixtureImpl = Generator[Tuple[popxl.tensor.Variable, popxl.Tensor], None, None]
_TestFixture = Callable[
    [np.ndarray, Optional[popxl.dtype], Optional[popxl.ReplicaGrouping]],
    _TestFixtureImpl,
]


def onchip_var(
    w_data: np.ndarray,
    dtype: Optional[popxl.dtype] = None,
    replica_grouping: Optional[popxl.ReplicaGrouping] = None,
) -> _TestFixtureImpl:
    w = popxl.variable(w_data, name="w", dtype=dtype, replica_grouping=replica_grouping)
    yield w, w


def remote_var(
    w_data: np.ndarray,
    dtype: Optional[popxl.dtype] = None,
    replica_grouping: Optional[popxl.ReplicaGrouping] = None,
) -> _TestFixtureImpl:
    if replica_grouping is not None and replica_grouping.num_groups > 1:
        rb_shape = w_data.shape[1:]
    else:
        rb_shape = w_data.shape

    rb = popxl.remote_buffer(rb_shape, popxl.dtypes.dtype.as_dtype(w_data.dtype))
    w_var = popxl.remote_variable(
        w_data, rb, 0, name="w_var", dtype=dtype, replica_grouping=replica_grouping
    )
    w = popxl.ops.remote_load(rb, 0)

    yield w_var, w

    popxl.ops.remote_store(rb, 0, w)


def remote_replica_sharded_var(
    w_data: np.ndarray,
    dtype: Optional[popxl.dtype] = None,
    replica_grouping: Optional[popxl.ReplicaGrouping] = None,
) -> _TestFixtureImpl:
    shard_over = (
        replica_grouping.const_rg.group_size if replica_grouping is not None else None
    )
    collective_rg = replica_grouping.const_rg if replica_grouping is not None else None

    rb = popxl.replica_sharded_buffer(
        w_data.shape,
        popxl.dtypes.dtype.as_dtype(w_data.dtype),
        replica_grouping=replica_grouping,
        shard_over=shard_over,
    )
    w_var = popxl.remote_replica_sharded_variable(
        w_data, rb, 0, name="w_var", dtype=dtype, replica_grouping=replica_grouping
    )
    w = popxl.ops.remote_load(rb, 0)

    yield w_var, w

    # See T67437: Must have all-gather or scatter-reduce on an RTS tensor.
    w_gathered = popxl.ops.collectives.replicated_all_gather(w, group=collective_rg)
    w = popxl.ops.collectives.replica_sharded_slice(w_gathered, group=collective_rg)
    popxl.ops.remote_store(rb, 0, w)


def replica_sharded_var(
    w_data: np.ndarray,
    dtype: Optional[popxl.dtype] = None,
    replica_grouping: Optional[popxl.ReplicaGrouping] = None,
) -> _TestFixtureImpl:
    shard_over = (
        replica_grouping.const_rg.group_size if replica_grouping is not None else None
    )
    collective_rg = replica_grouping.const_rg if replica_grouping is not None else None

    w_var, w = popxl.replica_sharded_variable(
        w_data,
        name="w_var",
        dtype=dtype,
        replica_grouping=replica_grouping,
        shard_over=shard_over,
    )

    yield w_var, w

    # See T67437: Must have all-gather or scatter-reduce on an RTS tensor.
    w_gathered = popxl.ops.collectives.replicated_all_gather(w, group=collective_rg)
    w = popxl.ops.collectives.replica_sharded_slice(w_gathered, group=collective_rg)


@pytest.mark.parametrize(
    "create_var_fixture",
    [onchip_var, remote_var, remote_replica_sharded_var, replica_sharded_var],
)
def test_mmap_variable(
    tmpdir: pathlib.Path,
    create_var_fixture: _TestFixture,
):
    """
    Test creating a Variable using an np.memmap:

    Test that weights_to_host() should call `flush` on the np.memmap object
    we passed during Ir construction.

    Test that sess.get_tensor_data(w) returns the exact same np.memmap object we
    passed during Ir construction.

    Test that the array returned by sess.get_tensor_data(w) is equal to the
    expected value for the computation performed.
    """

    ir = popxl.Ir(replication=2)

    # Numpy bug? Doesn't work if pass a Path object, but works with string.
    w_filepath = str(tmpdir / "w.npy")

    w_data = np.memmap(w_filepath, dtype=np.float32, shape=(20, 20), mode="w+")
    w_data.fill(1.0)

    expected_final_w_data = w_data + 1

    create_var_f = create_var_fixture(w_data)

    with ir.main_graph, popxl.in_sequence():
        # Setup (create w_var and w)
        w_var, w = next(create_var_f)
        w += 1
        # Teardown (remote_store). There may be no teardown, this is OK.
        try:
            next(create_var_f)
        except StopIteration:
            pass

    w_data_flush = w_data.flush
    flushed = False

    # pylint: disable=unused-argument
    def np_array_flush_wrapper(self: np.memmap, *args, **kwargs):
        nonlocal flushed
        flushed = True
        # Note, no need to pass `self`, w_data_flush is already bound to
        # w_data.
        return w_data_flush(*args, **kwargs)

    with mk_session_with_test_device(ir) as sess:
        sess.run()

        # Monkeypatch right before weights_to_host
        w_data.flush = MethodType(np_array_flush_wrapper, w_data)

    # Restore original method after weights_to_host
    w_data.flush = w_data_flush

    assert (
        flushed
    ), "weights_to_host() should have resulted in flushing the np.memmap used as the variable data"

    actual_updated_w = sess.get_tensor_data(w_var)

    assert np.array_equal(
        actual_updated_w, expected_final_w_data
    ), "Actual updated w does not match expected value"

    assert (
        actual_updated_w is w_data
    ), "sess.get_tensor_data(w) should return the exact same np.memmap object we passed during Ir construction"


"""
In the following error-case tests, we use the fixtures, but only use the setup
stage to create the variables; we do not need to remote_store.
"""


@pytest.mark.parametrize(
    "create_var_fixture",
    [onchip_var, remote_var, remote_replica_sharded_var, replica_sharded_var],
)
def test_conflicting_dtype_throws(
    tmpdir: pathlib.Path, create_var_fixture: _TestFixture
):
    ir = popxl.Ir(replication=2)

    # Numpy bug? Doesn't work if pass a Path object, but works with string.
    w_filepath = str(tmpdir / "w.npy")

    w_data = np.memmap(w_filepath, dtype=np.float32, shape=(4,), mode="w+")

    with ir.main_graph:
        with pytest.raises(ValueError) as exc:
            next(create_var_fixture(w_data, dtype=popxl.dtypes.half))
        assert exc.value.args[0].startswith(
            "When passing a memory-mapped NumPy array, the `dtype` parameter"
        )


@pytest.mark.parametrize(
    "create_var_fixture",
    [onchip_var, remote_var, remote_replica_sharded_var, replica_sharded_var],
)
def test_not_c_array_throws(tmpdir: pathlib.Path, create_var_fixture: _TestFixture):
    ir = popxl.Ir(replication=2)

    # Numpy bug? Doesn't work if pass a Path object, but works with string.
    w_filepath = str(tmpdir / "w.npy")

    # Note order='F', and needs to be more than rank 1 for order to have an
    # effect.
    w_data = np.memmap(w_filepath, dtype=np.float32, shape=(4, 4), mode="w+", order="F")

    with ir.main_graph:
        with pytest.raises(ValueError) as exc:
            next(create_var_fixture(w_data))
        assert exc.value.args[0].startswith(
            "When passing a memory-mapped NumPy array, it must already be in C-ordered"
        )


@pytest.mark.parametrize(
    "create_var_fixture",
    [onchip_var, remote_var, remote_replica_sharded_var, replica_sharded_var],
)
def test_requires_downcasting_throws(
    tmpdir: pathlib.Path, create_var_fixture: _TestFixture
):
    ir = popxl.Ir(replication=2)

    # Numpy bug? Doesn't work if pass a Path object, but works with string.
    w_filepath = str(tmpdir / "w.npy")

    # Note bad dtype
    w_data = np.memmap(w_filepath, dtype=np.double, shape=(4,), mode="w+")

    def test(bad_dtype):
        with pytest.raises(ValueError) as exc:
            next(create_var_fixture(w_data, dtype=bad_dtype))
        assert exc.value.args[0].startswith(
            f"When passing a memory-mapped NumPy array, the dtype {bad_dtype} "
            "is not supported, as it requires downcasting"
        )

    with ir.main_graph:
        # Test when dtype left to default (infer from w_data)
        test(None)
        # Test when dtype explicitly passed.
        test(popxl.dtypes.double)


@pytest.mark.parametrize(
    "create_var_fixture",
    [onchip_var, remote_var, remote_replica_sharded_var, replica_sharded_var],
)
def test_non_const_rg_warns(
    tmpdir: pathlib.Path,
    capfd: pytest.CaptureFixture[str],
    create_var_fixture: _TestFixture,
):
    """
    Test that a warning is emitted when you try to combine an np.memmap
    with a non-const replica grouping.
    """

    ir = popxl.Ir(replication=8)

    rg = ir.replica_grouping_from_assignments([0, 0, 1, 1, 0, 0, 1, 1])
    assert not rg.is_const, "Bad test: expected non-const replica grouping"

    # Numpy bug? Doesn't work if pass a Path object, but works with string.
    w_filepath = str(tmpdir / "w.npy")
    w_data = np.memmap(
        w_filepath,
        dtype=np.float32,
        shape=(
            rg.num_groups,
            rg.group_size * 2,
        ),
        mode="w+",
    )

    def clear_log():
        capfd.readouterr()

    with ir.main_graph:
        clear_log()
        next(create_var_fixture(w_data, replica_grouping=rg))
        log = capfd.readouterr().err

    import re

    # Comes from popxl.variable in tensor.py
    warn_msg = re.escape(
        "you have passed np.memmap data and a non-const replica grouping"
    )
    pattern = re.compile(warn_msg)
    matches = re.findall(pattern, log)

    assert (
        len(matches) == 1
    ), f"Failed to find warning message for non-const replica grouping in log. Log was:\n{log}"


@pytest.mark.parametrize(
    "create_var_fixture",
    [onchip_var, remote_var, remote_replica_sharded_var, replica_sharded_var],
)
def test_non_const_rg_runs_without_error(
    tmpdir: pathlib.Path,
    create_var_fixture: _TestFixture,
):
    """
    When passing a non-const replica grouping with an np.memmap, PopXL will fall
    back to treating the data as not an mmap.

    We test that the rest of the workflow (compile, run, etc) runs correctly and
    without error.
    """

    ir = popxl.Ir(replication=8)

    rg = ir.replica_grouping_from_assignments([0, 0, 1, 1, 0, 0, 1, 1])
    assert not rg.is_const, "Bad test: expected non-const replica grouping"

    # Numpy bug? Doesn't work if pass a Path object, but works with string.
    w_filepath = str(tmpdir / "w.npy")
    w_data = np.memmap(
        w_filepath,
        dtype=np.float32,
        shape=(
            rg.num_groups,
            rg.group_size * 2,
        ),
        mode="w+",
    )
    w_data.fill(1.0)

    expected_final_w_data = w_data + 1

    create_var_f = create_var_fixture(w_data, replica_grouping=rg)

    with ir.main_graph, popxl.in_sequence():
        # Setup (create w_var and w)
        w_var, w = next(create_var_f)
        w += 1
        # Teardown. There may be no teardown, this is OK.
        try:
            next(create_var_f)
        except StopIteration:
            pass

    with mk_session_with_test_device(ir) as sess:
        sess.run()

    actual_updated_w = sess.get_tensor_data(w_var)

    assert np.array_equal(
        actual_updated_w, expected_final_w_data
    ), "Actual updated w does not match expected value"
