# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import numpy as np
import pytest
import torch

import popxl
from popxl.testing import get_all_dtypes, get_all_int_dtypes
from popxl.dtypes import _PT_TO_POPXL, _NP_TO_POPXL


class Testdtype:
    def test_constructor(self):
        with pytest.raises(TypeError):
            popxl.dtype()

    def test_properties(self):
        dtypes = get_all_dtypes()
        uint_dtypes = get_all_int_dtypes(include_signed=False)
        int_dtypes = get_all_int_dtypes()
        for popxl_dtype in dtypes:
            if popxl_dtype in uint_dtypes:
                # PyTorch doesn't have unsigned integers. These are tested
                # below.
                continue
            torch_dtype = eval(f"torch.{popxl_dtype.name}")
            assert torch_dtype.is_complex == popxl_dtype.is_complex
            assert torch_dtype.is_floating_point == popxl_dtype.is_floating_point
            assert torch_dtype.is_signed == popxl_dtype.is_signed
            assert popxl_dtype.is_int == (popxl_dtype in int_dtypes)
        for popxl_dtype in uint_dtypes:
            # pylint: disable=unsubscriptable-object
            torch_dtype = eval(f"torch.{popxl_dtype.name[1:]}")
            assert torch_dtype.is_complex == popxl_dtype.is_complex
            assert torch_dtype.is_floating_point == popxl_dtype.is_floating_point
            assert torch_dtype.is_signed is True
            assert popxl_dtype.is_int is True

    def test_aliases(self):
        assert popxl.half == popxl.float16
        assert popxl.float == popxl.float32
        assert popxl.double == popxl.float64

    def test_conversion_numpy(self):
        popxl_dtypes = get_all_dtypes()
        np_dtypes = [eval(f"np.{popxl_dtype.name}") for popxl_dtype in popxl_dtypes]

        for popxl_dtype, np_dtype in zip(popxl_dtypes, np_dtypes):
            arr = np.zeros((1,), np_dtype)
            assert popxl_dtype == popxl.dtype.as_dtype(arr)
            assert popxl_dtype == popxl.dtype.as_dtype(np_dtype)
            assert popxl_dtype.as_numpy() == np_dtype

        with pytest.raises(ValueError):
            popxl.dtype.as_dtype(np.str)

    def test_conversion_string(self):
        popxl_dtypes = get_all_dtypes()
        names = [popxl_dtype.name for popxl_dtype in popxl_dtypes]

        for popxl_dtype, name in zip(popxl_dtypes, names):
            assert popxl_dtype == popxl.dtype.as_dtype(name)

    def test_conversion_python(self):
        import builtins

        py_to_pir = {
            builtins.bool: popxl.bool,
            builtins.float: popxl.float32,
            builtins.int: popxl.int64,
        }

        for py_type, popxl_dtype in py_to_pir.items():
            assert popxl_dtype == popxl.dtype.as_dtype(py_type)

    def test_conversion_torch(self):
        assert len(_PT_TO_POPXL) > 0

        # Dict of NumPy dtype -> torch dtype (when the correspondence exists)
        # Adapted from torch internals
        numpy_to_torch_dtype_dict = {
            np.dtype("bool"): torch.bool,
            np.dtype("uint8"): torch.uint8,
            np.dtype("int8"): torch.int8,
            np.dtype("int16"): torch.int16,
            np.dtype("int32"): torch.int32,
            np.dtype("int64"): torch.int64,
            np.dtype("float16"): torch.float16,
            np.dtype("float32"): torch.float32,
            np.dtype("float64"): torch.float64,
            np.dtype("complex64"): torch.complex64,
            np.dtype("complex128"): torch.complex128,
        }

        torch_to_np = {t: np for np, t in numpy_to_torch_dtype_dict.items()}

        for torch_type, popxl_type in _PT_TO_POPXL.items():
            np_type = torch_to_np[torch_type]
            popxl_type_via_torch = _NP_TO_POPXL[np_type]
            assert popxl_type_via_torch == popxl_type
