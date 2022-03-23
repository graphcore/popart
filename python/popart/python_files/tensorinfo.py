# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import numpy as np
from popart_core import _TensorInfoCore
from typing import List, Union, Iterable, Tuple


def _get_popart_type(dtype: np.dtype) -> str:
    """Return the relevant PopART type string from a numpy dtype.

    Arguments:
        dtype: numpy dtype

    Returns:
        PopART dtype string.
    """
    return {
        np.uint8: 'UINT8',
        np.uint16: 'UINT16',
        np.uint32: 'UINT32',
        np.uint64: 'UINT64',
        np.int8: 'INT8',
        np.int16: 'INT16',
        np.int32: 'INT32',
        np.int64: 'INT64',
        np.float16: 'FLOAT16',
        np.float32: 'FLOAT',
        np.float64: 'DOUBLE',
        np.bool_: 'BOOL'
    }[dtype]


class TensorInfo(_TensorInfoCore):
    """Python wrapper to ``TensorInfo`` to handle numpy types in constructor.

    For example:
        TensorInfo(dtype, shape)

        TensorInfo(numpy.ndarray)

    Raises:
        TypeError: Raised if incorrect type is used to create a tensorinfo.
    """

    def __init__(self, *args: Union[Iterable, np.array]) -> None:
        def unpack_args() -> Tuple[np.dtype, List[int]]:
            """Return (dtype, shape) from *args.

            Raises:
                TypeError: Raised if incorrect type is used to create a tensorinfo.

            Returns:
                Tuple of numpy dtype and tensor shape.
            """
            if len(args) == 2:
                return args
            elif len(args) == 1 and isinstance(args[0], np.ndarray):
                a = args[0]
                return (a.dtype, a.shape)
            else:
                raise TypeError(f'Can not create TensorInfo with args {args}')

        def sanitize_dtype(dtype: Union[type, np.dtype]) -> str:
            """Convert a ``type`` or ``numpy.dtype`` to a ``str``.

            Arguments:
                dtype: Python type (for example, ``float``) or numpy dtype
                (for example, ``np.float32``) to convert to PopART type string.

            Returns:
                PopART type string.
            """
            if isinstance(dtype, np.dtype):
                return _get_popart_type(dtype.type)
            elif isinstance(dtype, type):
                return _get_popart_type(dtype)
            else:
                return dtype

        dtype, shape = unpack_args()
        dtype = sanitize_dtype(dtype)

        super(TensorInfo, self).__init__(dtype, shape)
