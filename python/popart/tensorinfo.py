import popart
import numpy as np


def _get_popart_type(dtype):
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


class TensorInfo(popart.TensorInfoCore):
    """TensorInfo(dtype, shape)
    TensorInfo(numpy.ndarray)"""

    def __init__(self, *args):
        # Return (dtype, shape) from *args
        def unpack_args():
            if len(args) == 2:
                return args
            elif len(args) == 1 and isinstance(args[0], np.ndarray):
                a = args[0]
                return (a.dtype, a.shape)
            else:
                raise TypeError(f'Can not create TensorInfo with args {args}')

        # Convert a `type` or `numpy.dtype` to a `str`
        def sanitize_dtype(dtype):
            if isinstance(dtype, np.dtype):
                return _get_popart_type(dtype.type)
            elif isinstance(dtype, type):
                return _get_popart_type(dtype)
            else:
                return dtype

        dtype, shape = unpack_args()
        dtype = sanitize_dtype(dtype)

        super(TensorInfo, self).__init__(dtype, shape)
