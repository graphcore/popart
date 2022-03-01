# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
"""`dtype`-related functions and classes that are useful during testing."""

import popxl

__all__ = ['get_all_dtypes', 'get_all_int_dtypes']

_all_signed_int_dtypes = [popxl.int8, popxl.int16, popxl.int32, popxl.int64]
_all_unsigned_int_dtypes = [
    popxl.uint8, popxl.uint16, popxl.uint32, popxl.uint64
]
_all_fp_dtypes = [popxl.float16, popxl.float32, popxl.float64]
_all_complex_dtypes = [popxl.complex64, popxl.complex128]


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
