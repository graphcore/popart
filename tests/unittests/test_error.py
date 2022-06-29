# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import pytest

import popart
import popart_core


def test_popart_error():
    with pytest.raises(popart.popart_exception):
        popart_core._throw_popart_error("")


def test_popart_internal_error():
    with pytest.raises(popart.popart_internal_exception):
        popart_core._throw_popart_internal_error("")

    # Can catch popart_internal_error as popart_exception
    with pytest.raises(popart.popart_exception):
        popart_core._throw_popart_internal_error("")


def test_popart_runtime_error():
    with pytest.raises(popart.popart_runtime_error):
        popart_core._throw_popart_runtime_error("")

    # Can catch popart_runtime_error as popart_exception
    with pytest.raises(popart.popart_exception):
        popart_core._throw_popart_runtime_error("")


def test_poplibs_error():
    with pytest.raises(popart.poplibs_exception):
        popart_core._throw_poplibs_error("")


def test_poplar_error():
    with pytest.raises(popart.poplar_exception):
        popart_core._throw_poplar_error("")


def test_poplar_runtime_error():
    with pytest.raises(popart.poplar_runtime_error):
        popart_core._throw_poplar_runtime_error("")


def test_application_runtime_error():
    with pytest.raises(popart.poplar_application_runtime_error):
        popart_core._throw_application_runtime_error("")

    # application_runtime_error can be caught as runtime_error
    with pytest.raises(popart.poplar_runtime_error):
        popart_core._throw_application_runtime_error("")


def test_system_runtime_error():
    with pytest.raises(popart.poplar_system_runtime_error):
        popart_core._throw_system_runtime_error("")

    # system_runtime_error can be caught as runtime_error
    with pytest.raises(popart.poplar_runtime_error):
        popart_core._throw_system_runtime_error("")


def test_recoverable_runtime_error():
    with pytest.raises(popart.poplar_recoverable_runtime_error) as e_info:
        popart_core._throw_recoverable_runtime_error("")
    assert e_info.value.recoveryAction == popart.RecoveryAction.IPU_RESET

    # recoverable_runtime_error can be caught as system_runtime_error
    with pytest.raises(popart.poplar_system_runtime_error):
        popart_core._throw_recoverable_runtime_error("")


def test_unrecoverable_runtime_error():
    with pytest.raises(popart.poplar_unrecoverable_runtime_error):
        popart_core._throw_unrecoverable_runtime_error("")

    # unrecoverable_runtime_error can be caught as system_runtime_error
    with pytest.raises(popart.poplar_system_runtime_error):
        popart_core._throw_unrecoverable_runtime_error("")


def test_unknown_runtime_error():
    with pytest.raises(popart.poplar_unknown_runtime_error):
        popart_core._throw_unknown_runtime_error("")

    # unknown_runtime_error can be caught as system_runtime_error
    with pytest.raises(popart.poplar_system_runtime_error):
        popart_core._throw_unknown_runtime_error("")
