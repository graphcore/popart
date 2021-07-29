# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
"""`dtype`-related functions and classes that are useful during testing."""

import popart.ir as pir

__all__ = ['get_all_dtypes', 'get_all_int_dtypes']

_all_signed_int_dtypes = [pir.int8, pir.int16, pir.int32, pir.int64]
_all_unsigned_int_dtypes = [pir.uint8, pir.uint16, pir.uint32, pir.uint64]
_all_fp_dtypes = [pir.float16, pir.float32, pir.float64]
_all_complex_dtypes = [pir.complex64, pir.complex128]


def get_all_dtypes():
    all_dtypes = []
    all_dtypes += get_all_int_dtypes()
    all_dtypes += _all_fp_dtypes
    all_dtypes += _all_complex_dtypes
    return all_dtypes


def get_all_int_dtypes(include_signed=True, include_unsigned=True):
    dtypes = []
    if include_signed:
        dtypes += _all_signed_int_dtypes
    if include_unsigned:
        dtypes += _all_unsigned_int_dtypes
    return dtypes
