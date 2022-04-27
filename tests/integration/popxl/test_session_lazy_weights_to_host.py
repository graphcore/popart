# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import pytest

import popxl
import popxl.ops as ops

from popxl_test_device_helpers import attach, mk_session_with_test_device


@pytest.fixture(scope='module', autouse=True)
def capture_popart_logging():
    from unittest.mock import patch
    with patch.dict("os.environ", {
            "POPART_LOG_LEVEL": "DEBUG",
    }):
        yield


def test_get_tensor_data_elides_weights_to_host_if_host_weights_are_in_sync(
        capfd):
    """
    Test get_tensor_data will not result in a run of the WeightsToHost program
    if the host weights are currently in sync.

    That is, if no program has been run since the last get_tensor_data call. 
    """

    import re

    def assert_num_weights_to_host(num, msg):
        # This is very brittle
        weights_to_host_log_str = 'Writing weights to host complete.'
        pattern = re.compile(weights_to_host_log_str)

        log = capfd.readouterr().err

        # Go to start of file
        matches = re.findall(pattern, log)
        assert len(matches) == num, f'{msg}\nLog was:\n{log}'

    def clear_log():
        capfd.readouterr()

    ir = popxl.Ir()
    mg = ir.main_graph

    with mg:
        c = popxl.constant(1.)
        w = popxl.variable(2., name='w')
        v = popxl.variable(3., name='v')

        ops.var_updates.accumulate_(w, c)
        ops.var_updates.accumulate_(v, c)

    session: popxl.Session = mk_session_with_test_device(ir)

    # Clear log up to now, so we only inspect logs from the following runtime
    # functions under test.
    clear_log()

    with session:
        session.get_tensor_data(w)  # cache hit
        session.get_tensor_data(w)  # cache hit
        session.get_tensor_data(v)  # cache hit
        assert_num_weights_to_host(
            0,
            "Expected zero WeightsToHost when calling get_tensor_data immediately after construction."
        )

        session.write_variables_data({w: 2., v: 26.})
        session.get_tensors_data([w, v])  # hit
        session.get_tensor_data(w)  # hit
        session.get_tensor_data(v)  # hit
        assert_num_weights_to_host(
            0,
            "Expected zero WeightsToHost when calling get_tensor_data repeatedly after write_variables_data."
        )

        session.run()
        assert session.device.isAttached
        session.get_tensor_data(v)  # miss
        session.get_tensor_data(w)  # hit
        session.get_tensor_data(v)  # hit
        assert_num_weights_to_host(
            1,
            "Expected exactly 1 WeightsToHost when calling get_tensor_data repeatedly after run."
        )

    # As exiting ctxt caused a weights_to_host.
    clear_log()

    # Test get_tensor_data will get latest weights outside of context IF
    # manually attached.
    attach(session.device)
    # Note using device ctxt to ensure detach, not full Session ctxt.
    # Note device ctxtmgr does not attach, only detach on exit.
    with session.device:
        session.run()
        assert session.device.isAttached
        session.get_tensor_data(v)  # miss
        session.get_tensors_data([w, v])  # hit
        assert_num_weights_to_host(
            1,
            "Expected exactly 1 WeightsToHost when calling get_tensor_data repeatedly after run outside of the context but with manual attach."
        )
