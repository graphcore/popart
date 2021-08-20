# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from abc import ABC
from typing import Optional, Tuple, Union

import numpy as np

import popart._internal.ir as _ir
from popart.ir import dtypes
from popart.ir.globals import gcg

__all__ = ['Tensor', 'Placeholder', 'Constant', 'Variable']


class Tensor(ABC):
    def __init__(self, dtype: dtypes.dtype, shape: Tuple[int], name: str,
                 pb_tensor_id: str):
        """An abstract base class that represents a tensor in the PopART IR.

        Args:
            dtype (dtype):
                The data type of the tensor.
            shape (Tuple[int]):
                The shape of the tensor.
            name (str):
                The name of the tensor.
            pb_tensor_id (str):
                The ID of the corresponding pybind11 tensor.
        """
        self._dtype = dtype
        self._shape = shape
        self._name = name
        self._pb_tensor_id = pb_tensor_id
        self._graph = gcg()

    @property
    def dim(self) -> int:
        """Returns the number of dimensions of `self` tensor.

        Returns:
            int: The number of dimensions of `self` tensor.
        """
        return len(self._shape)

    @property
    def dtype(self) -> dtypes.dtype:
        return self._dtype

    @property
    def shape(self) -> Tuple[int]:
        return self._shape

    @property
    def nelms(self) -> int:
        return 1 if self._shape == () else sum(self._shape)

    @property
    def name(self) -> str:
        return self._name

    def _ensure_tensor(self, other) -> 'Tensor':
        """A helper method that's used in operator overloading to ensure that
        all operands are of type `Tensor`.

        If any they are not, an attempt is made to convert the operands to a
        `Constant` tensor.

        Returns:
            Tensor:
                A `popart.ir.Tensor`.
        """
        if isinstance(other, Tensor):
            return other
        else:
            return Constant(other, self._dtype)

    def __repr__(self) -> str:
        return self._name


class Placeholder(Tensor):
    def __init__(self,
                 dtype: dtypes.dtype,
                 shape: Tuple[int],
                 name: Optional[str] = None):
        """A placeholder `Tensor` used to represent a place in the graph that
        will be filled with data during runtime.

        This class only holds metadata for this place - its data type and its
        shape. This could be, for example, the input of a subraph or a tensor
        that has been produced by an operation.

        Args:
            dtype (dtype):
                The data type of the tensor.
            shape (Tuple[int]):
                The shape of the tensor.
            name (Optional[str]):
                The name of the tensor. Defaults to None.
        """
        g = gcg()
        pb_g = g._pb_graph

        name = name if name else 't'
        name = g._create_tensor_name(name)
        pb_tensor_id = pb_g.addScope(name)

        super().__init__(dtype, shape, name, pb_tensor_id)


class Variable(Tensor):
    def __init__(self,
                 data: np.array,
                 dtype: Optional[dtypes.dtype] = dtypes.float32,
                 name: Optional[str] = None):
        """A variable tensor is initialised with data during graph creation.

        This tensor can be used to represent a model weight or any other
        parameter that can change while running a model.

        Args:
            data (np.array):
                The data used to initialise the tensor.
            dtype (dtype):
                The data type of the tensor. Defaults to `pir.float32`.
            name (Optional[str]):
                The name of the tensor. Defaults to `None`.
        """
        g = gcg()
        pb_g = g._pb_graph

        data = np.array(data, dtype=dtype.as_numpy())

        name = name if name else 't'
        name = g._create_tensor_name(name)
        info = _ir.TensorInfo(
            dtypes.dtype.as_dtype(data)._pb_dtype, data.shape)
        pb_tensor_id = pb_g.addScope(name)
        pb_g.addVarInit(pb_tensor_id, info, data)

        super().__init__(dtypes.dtype.as_dtype(data), data.shape, name,
                         pb_tensor_id)


class Constant(Tensor):
    def __init__(self,
                 data: Union[np.array, int, tuple, list],
                 dtype: Optional[dtypes.dtype] = dtypes.float32,
                 name: Optional[str] = None):
        """A constant tensor is initialised with data during graph creation.

        This tensor cannot change during the runtime of a model. The inteded use
        of this class is when doing operations between `popart.ir.Tensor`
        instances and other types, such as `numpy.ndarray` objects, numbers, or
        list or tuples of numbers.

        Example:
            >>> import popart.ir as pir
            >>> main = pir.Ir().main_graph()
            >>> with main:
            >>>     a = pir.Variable(0)
            >>>     # The `1` will be implicitly converted to a `Constant`.
            >>>     b = a + 1

        Args:
            data (np.array):
                The data used to initialise the tensor.
            dtype (dtype):
                The data type of the tensor. Defaults to `pir.float32`.
            name (Optional[str]):
                The name of the tensor. Defaults to `None`.
        """
        g = gcg()
        pb_g = g._pb_graph

        data = np.array(data, dtype=dtype.as_numpy())

        name = name if name else 't'
        name = g._create_tensor_name(name)
        info = _ir.TensorInfo(
            dtypes.dtype.as_dtype(data)._pb_dtype, data.shape)
        pb_tensor_id = pb_g.addScope(name)
        pb_g.addConstInit(pb_tensor_id, info, data)

        super().__init__(dtypes.dtype.as_dtype(data), data.shape, name,
                         pb_tensor_id)
