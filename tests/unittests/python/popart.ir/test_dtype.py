# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import pytest
import torch  # pylint: disable=unused-import

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
        err_msg = "Cannot create popart.ir.dtype.dtype instances."
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
