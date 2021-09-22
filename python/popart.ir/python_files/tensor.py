# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np

import popart._internal.ir as _ir
from popart.ir import dtypes
from popart.ir.globals import gcg

__all__ = ['Tensor', 'Variable', 'variable', 'Constant', 'constant']


class Tensor:
    def __init__(self):
        """Wraps a tensor in the PopART IR."""
        self._pb_tensor: _ir.Tensor
        raise RuntimeError("pir.Tensor cannot be constructed directly.")

    @classmethod
    def _from_pb_tensor(cls, pb_tensor: _ir.Tensor) -> 'Tensor':
        self = super().__new__(cls)
        self._pb_tensor = pb_tensor
        return self

    @property
    def id(self) -> str:
        return str(self._pb_tensor.id)

    @property
    def dtype(self) -> dtypes.dtype:
        return dtypes.dtype.as_dtype(self._pb_tensor.info.dataType())

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self._pb_tensor.info.shape())

    @property
    def rank(self) -> int:
        return self._pb_tensor.info.rank()

    @property
    def nelms(self) -> int:
        return self._pb_tensor.info.nelms()

    @property
    def name(self) -> str:
        return _ir.removeScope(self._pb_tensor.getGraph(), self.id)

    def __str__(self) -> str:
        return f"{self.name} {self.dtype} {self.shape}"

    def _ensure_tensor(self, value: Any,
                       dtype: Optional[dtypes.dtype] = None) -> 'Tensor':
        """A helper method that's used in operator overloading to ensure that
        all operands are of type `Tensor`.

        If any they are not, an attempt is made to convert the operands to a
        `Constant` tensor.

        Returns:
            Tensor:
                A `popart.ir.Tensor`.
        """
        if isinstance(value, Tensor):
            return value
        else:
            dtype = self.dtype if dtype is None else dtype
            return constant(value, dtype)

    def __add__(self, value: Any) -> 'Tensor':
        """Returns ops.add(self, value)."""
        import popart.ir.ops as ops
        return ops.add(self, self._ensure_tensor(value))

    def __sub__(self, value: Any) -> 'Tensor':
        """Returns ops.sub(self, value)."""
        import popart.ir.ops as ops
        return ops.sub(self, self._ensure_tensor(value))

    def __mul__(self, value: Any) -> 'Tensor':
        """Returns ops.mul(self, value)."""
        import popart.ir.ops as ops
        return ops.mul(self, self._ensure_tensor(value))

    def __truediv__(self, value: Any) -> 'Tensor':
        """Returns ops.div(self, value)."""
        import popart.ir.ops as ops
        return ops.div(self, self._ensure_tensor(value))

    def transpose(self,
                  permutation: Optional[Tuple[int, ...]] = None) -> 'Tensor':
        """Returns ops.transpose(self, permutation)."""
        import popart.ir.ops as ops
        return ops.transpose(self, permutation)

    def reshape(self, shape: Tuple[int, ...]) -> 'Tensor':
        """Returns ops.reshape(self, shape)."""
        import popart.ir.ops as ops
        return ops.reshape(self, shape)

    @property
    def T(self) -> 'Tensor':
        """Returns the Tensor transposed with reversed axes."""
        return self.transpose()


class Variable(Tensor):
    """Wraps a Tensor in the PopART IR that has TensorType.Variable"""


class Constant(Tensor):
    """Wraps a Tensor in the PopART IR that has TensorType.Constant"""


def variable(data: Union[np.ndarray, Sequence[Any], int, float],
             dtype: Optional[dtypes.dtype] = dtypes.float32,
             name: Optional[str] = None) -> Variable:
    """A variable tensor that is initialised with data during graph creation.

    This tensor can be used to represent a model weight or any other
    parameter that can change while running a model.

    Must be created in the main graph scope. Example:
        >>> import popart.ir as pir
        >>> with pir.Ir().main_graph():
        >>>     a = pir.variable(0)

    Args:
        data (np.ndarray, or a value numpy can use to construct an np.ndarray):
            The data used to initialise the tensor.
        dtype (dtype):
            The data type of the tensor. Defaults to `pir.float32`.
        name (Optional[str]):
            The name of the tensor. Defaults to `None`.
    """
    g = gcg()
    pb_g = g._pb_graph
    data = np.array(data, dtype=dtype.as_numpy())
    info = _ir.TensorInfo(dtype._pb_dtype, data.shape)
    pb_id = g._create_tensor_id(name)
    pb_g.addVarInit(pb_id, info, data)
    return Variable._from_pb_tensor(pb_g.getTensor(pb_id))


def constant(data: Union[np.ndarray, Sequence[Any], int, float],
             dtype: Optional[dtypes.dtype] = dtypes.float32,
             name: Optional[str] = None) -> Constant:
    """A constant tensor that is initialised with data during graph creation.

    This tensor cannot change during the runtime of a model. The inteded use
    of this class is when doing operations between `popart.ir.Tensor`
    instances and other types, such as `numpy.ndarray` objects, numbers, or
    list or tuples of numbers.

    Example:
        >>> import popart.ir as pir
        >>> with pir.Ir().main_graph():
        >>>     a = pir.variable(0)
        >>>     # The `1` will be implicitly converted to a `Constant`.
        >>>     b = a + 1

    Args:
        data (np.array, or a value numpy can use to construct an np.ndarray):
            The data used to initialise the tensor.
        dtype (dtype):
            The data type of the tensor. Defaults to `pir.float32`.
        name (Optional[str]):
            The name of the tensor. Defaults to `None`.
    """
    g = gcg()
    pb_g = g._pb_graph
    data = np.array(data, dtype=dtype.as_numpy())
    info = _ir.TensorInfo(dtype._pb_dtype, data.shape)
    pb_id = g._create_tensor_id(name)
    pb_g.addConstInit(pb_id, info, data)
    return Constant._from_pb_tensor(pb_g.getTensor(pb_id))
