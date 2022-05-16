# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import pytest

import popxl
import popxl.ops as ops
import numpy as np

from popxl_test_device_helpers import mk_session_with_test_device


def test_session_runtime_fns_guard_attached_behaviour():
    """
    Tests the runtime functions of Session either throw ValueError or do not
    throw (depending on which function) when not attached; and do not throw
    when attached.
    """
    ir = popxl.Ir()

    w_data = np.array(2, dtype=np.float32)

    with ir.main_graph:
        w = popxl.variable(w_data)
        y = popxl.constant(2)
        v = ops.var_updates.accumulate_(w, y)

    session: popxl.Session = mk_session_with_test_device(ir)

    require_attach_runtime_fns = [
        lambda s: s.run(), lambda s: s.run_with_outputs({}, {}), lambda s: s.
        weights_from_host(), lambda s: s.weights_to_host()
    ]
    can_be_detached_runtime_fns = [
        lambda s: s.write_variable_data(w, w_data), lambda s: s.
        write_variables_data({w: w_data}), lambda s: s.get_tensor_data(
            w), lambda s: s.get_tensors_data([w])
    ]

    def assert_all_fns_throw(fns, should_throw: bool):
        """
        If should_throw, assert all fns should throw ValueError, otherwise
        assert they do not throw.
        """
        for fn in fns:
            if should_throw:
                with pytest.raises(ValueError):
                    fn(session)
            else:
                fn(session)

    # Not yet attached
    assert_all_fns_throw(require_attach_runtime_fns, should_throw=True)
    assert_all_fns_throw(can_be_detached_runtime_fns, should_throw=False)

    # Can run outside context if manually attached
    with session:
        assert_all_fns_throw(require_attach_runtime_fns, should_throw=False)
        assert_all_fns_throw(can_be_detached_runtime_fns, should_throw=False)

    assert_all_fns_throw(require_attach_runtime_fns, should_throw=True)
    assert_all_fns_throw(can_be_detached_runtime_fns, should_throw=False)


def test_reentry():
    ir = popxl.Ir()

    with ir.main_graph:
        w = popxl.variable(1)
        y = popxl.constant(2)
        v = ops.var_updates.accumulate_(w, y)

    session: popxl.Session = mk_session_with_test_device(ir)

    for i in range(2):
        # attach, loadEngineAndConnectStreams, weightsFromHost
        with session:
            pass
        # weightsToHost, detach
        assert not session.is_attached


def test_session_ctxtmgr_attach_detach():
    ir = popxl.Ir()

    with ir.main_graph:
        w = popxl.variable(1)
        y = popxl.constant(2)
        v = ops.var_updates.accumulate_(w, y)

    session: popxl.Session = mk_session_with_test_device(ir)

    # Starts not attached
    assert not session.is_attached
    with session:
        # Attached on enter
        assert session.is_attached

        # Nested enter maintains attach
        with session:
            assert session.is_attached
        # Exiting nested session does not clobber outer context's attach
        assert session.is_attached

        # Test `run`
        session.run()
        assert session.is_attached

        # Test `weights_to_host`
        session.weights_to_host()
        assert session.is_attached

    # Exit ensures detach
    assert not session.is_attached


def test_session_ctxtmgr_does_weights_from_host_on_enter():
    """
    Test that entering the context does a `weights_from_host` for the user, so
    they can immediately call `run` and not get garbage.

    NOTE: We cannot check the value of a weight was set on device by doing
          `session.get_tenor_data(w)` (and no call to `run`) because this will 
          be a nop as it will detect the weights are not yet out of sync and do
          nothing.  
    """

    ir = popxl.Ir()
    MAGIC = 234.54

    with ir.main_graph:
        w = popxl.variable(MAGIC)

        exp_w_h2d = popxl.h2d_stream(w.shape, w.dtype)

        # Will be fed MAGIC
        exp_w = ops.host_load(exp_w_h2d)

        # We expect these to be true:
        # If weights_from_host occured as expected, will be the same.
        # If weights_from_host did not occur as expected, w will be garbage, so
        # will be different.
        res = ops.equal(w, exp_w)

        d2h_res = popxl.d2h_stream(res.shape, res.dtype)

        ops.host_store(d2h_res, res)

    session: popxl.Session = mk_session_with_test_device(ir)

    with session:
        # No explicit weightsFromHost here
        ins = {exp_w_h2d: MAGIC}
        outs = session.run(ins)

        assert outs[d2h_res]


def test_session_ctxtmgr_exit_weights_to_host_behaviour():
    """
    Test that on exiting of the context, a `weights_to_host` will implictly be
    done for the user; and that a subsequent call to `get_tensor_data(w)`
    outside of the context is a) permitted and b) will not attempt to attach to
    the device and re-fetch weights, but use the weights now on the host.

    We do this by updating a variable in the Ir, and checking that after the
    context `session.get_tensor_data(w)` returns the new value, instead of
    throwing due to needing to access the device as the weights are still out of
    sync but leaving the context caused a detach.
    """

    ir = popxl.Ir()

    MAGIC = np.array(345345.345, dtype=np.float32)

    with ir.main_graph:
        w = popxl.variable(np.array(1.0), dtype=popxl.dtypes.float32)

        x_h2d = popxl.h2d_stream(w.shape, w.dtype)
        x = ops.host_load(x_h2d)

        ops.var_updates.copy_var_update_(w, x)

    session: popxl.Session = mk_session_with_test_device(ir)

    with session:
        # We are not testing weights_from_host occured on ctxt entry, so
        # manually ensure it happens here to get a clearer error in this test
        # if that behaviour is broken too.
        session.weights_from_host()

        ins = {x_h2d: MAGIC}
        session.run(ins)

        # No explicit weights_to_host()

    assert not session.is_attached

    w_h = session.get_tensor_data(w)

    # Should not have attached due to get_tensor_data
    assert not session.is_attached

    # Should be the new value of w that was updated on device
    assert np.allclose(w_h, MAGIC)
