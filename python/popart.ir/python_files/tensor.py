from typing import List, Optional, Tuple, TypeVar, Union, TYPE_CHECKING, Type

import numpy as np

import popart._internal.ir as _ir
from popart.ir.globals import gcg
from popart.util.ast import parse_assign_from_ast

if TYPE_CHECKING:
    from popart.ir.graph import Graph

TensorLike = Union['Tensor', int]
Shape = Tuple[int]
Tensors = List['Tensor']


class TensorInfo:
    def __init__(self, dtype, shape: Shape) -> None:
        self._dtype = dtype
        self._shape = shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def shape(self):
        return self._shape


class Tensor:
    def __init__(
            self,
            dtype: 'dtype',
            shape: Shape,
            debug_name: Optional[str] = None,
    ):
        self._dtype = dtype
        self._shape = shape
        self._info = TensorInfo(dtype, shape)
        if debug_name is None:
            debug_name = parse_assign_from_ast()
        self._debug_name = debug_name
        self._graph: 'Graph' = gcg()
        self._tensor = self._graph._graph

    # def __add__(self, rhs: 'Tensor') -> 'Tensor':
    # return gcg().create_op()

    # def __radd__(self, lhs: TensorLike) -> 'Tensor':
    #     return lhs.__add__(self)

    # def __iadd__(self, rhs: TensorLike) -> 'Tensor':
    #     # return add(self, rhs, inplace=True)

    # # def __matmul__(self, rhs: TensorLike) -> 'Tensor':
    #     # raise matmul(self, rhs)

    # def __rmatmul__(self, lhs: TensorLike) -> 'Tensor':
    #     raise lhs.__matmul__(self)

    def __repr__(self) -> str:
        return self._debug_name

    def dim(self) -> int:
        """Returns the number of dimensions of `self` tensor.

        Returns:
            int: The number of dimensions of `self` tensor.
        """
        return len(self._shape)

    @property
    def dtype(self) -> 'dtype':
        return self._dtype

    @property
    def shape(self) -> Shape:
        return self._shape

    @property
    def nelms(self) -> int:
        return self._shape

    @property
    def info(self) -> TensorInfo:
        return self._info

    def get_graph(self) -> 'Graph':
        return self._graph
