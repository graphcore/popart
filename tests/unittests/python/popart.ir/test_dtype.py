# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import numpy as np
import pytest
import torch

import popart.ir as pir
from popart.ir.testing import get_all_dtypes, get_all_int_dtypes


# TODO(T38031): Delete this utility function.
def _get_torch_version():
    """Utility function to convert the torch version to a tuple of ints.

    Returns:
        tuple: The version - a tuple of ints.
    """
    v = torch.__version__
    v = v.split('+')[0]
    v = v.split('.')
    v = tuple([int(i) for i in v])
    return v


class Testdtype:
    def test_constructor(self):
        with pytest.raises(TypeError) as excinfo:
            pir.dtype()
        err_msg = "Cannot create popart.ir.dtypes.dtype instances."
        assert str(excinfo.value) == err_msg

    # TODO(T38031): Unskip this test.
    @pytest.mark.skipif(_get_torch_version() < (1, 7, 1),
                        reason="Requires torch>=1.7.1.")
    def test_properties(self):
        dtypes = get_all_dtypes()
        uint_dtypes = get_all_int_dtypes(include_signed=False)
        for pir_dtype in dtypes:
            if pir_dtype in uint_dtypes:
                # PyTorch doesn't have unsigned integers. These are tested
                # below.
                continue
            torch_dtype = eval(f'torch.{pir_dtype._name}')
            assert torch_dtype.is_complex == pir_dtype.is_complex
            assert torch_dtype.is_floating_point == pir_dtype.is_floating_point
            assert torch_dtype.is_signed == pir_dtype.is_signed
        for pir_dtype in uint_dtypes:
            torch_dtype = eval(f'torch.{pir_dtype._name[1:]}')
            assert torch_dtype.is_complex == pir_dtype.is_complex
            assert torch_dtype.is_floating_point == pir_dtype.is_floating_point
            assert torch_dtype.is_signed == True

    def test_aliases(self):
        assert pir.half == pir.float16
        assert pir.float == pir.float32
        assert pir.double == pir.float64

    def test_conversion_numpy(self):
        pir_dtypes = get_all_dtypes()
        np_dtypes = [eval(f'np.{pir_dtype._name}') for pir_dtype in pir_dtypes]

        for pir_dtype, np_dtype in zip(pir_dtypes, np_dtypes):
            arr = np.zeros((1, ), np_dtype)
            assert pir_dtype == pir.dtype.as_dtype(arr)
            assert pir_dtype == pir.dtype.as_dtype(np_dtype)
            assert pir_dtype.as_numpy() == np_dtype

        with pytest.raises(ValueError) as excinfo:
            pir.dtype.as_dtype(np.str)
        exp_message = (f'There is not a `popart.ir.dtype` that is '
                       f'compatible with {np.str}.')
        assert str(excinfo.value) == exp_message

    def test_conversion_string(self):
        pir_dtypes = get_all_dtypes()
        names = [pir_dtype._name for pir_dtype in pir_dtypes]

        for pir_dtype, name in zip(pir_dtypes, names):
            assert pir_dtype == pir.dtype.as_dtype(name)

    def test_conversion_python(self):
        import builtins
        py_to_pir = {builtins.bool: pir.bool, builtins.float: pir.float32}

        for py_type, pir_dtype in py_to_pir.items():
            assert pir_dtype == pir.dtype.as_dtype(py_type)

        with pytest.raises(ValueError) as excinfo:
            pir.dtype.as_dtype(builtins.int)
        exp_message = (f'There is not a `popart.ir.dtype` that is '
                       f'compatible with {builtins.int}.')
        assert str(excinfo.value) == exp_message
