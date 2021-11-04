# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Any, Dict, Iterable, Optional, Tuple, Type, Union
import numpy as np

import popart._internal.ir as _ir
from popart.ir import dtypes
from popart.ir.context import gcg
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from popart.ir import Ir

__all__ = [
    'Tensor', 'variable', 'constant', 'subgraph_input', 'subgraph_output'
]


class Tensor:
    def __init__(self):
        """popart.ir Tensor."""
        self._pb_tensor: _ir.Tensor
        raise RuntimeError("pir.Tensor cannot be constructed directly.")

    # Dictionary to track Tensor subclasses
    _tensor_types: Dict[str, 'Type[Tensor]'] = {}

    def __init_subclass__(cls, tensor_type: Optional[str] = None,
                          **kwargs) -> None:
        """Hook called when creating a Tensor subclass.
            Argument `tensor_type` is used to allow `_from_pb_tensor` to return
            the correct subclass for any Tensor retrieved from the internal IR"""
        super().__init_subclass__(**kwargs)
        if tensor_type is not None:
            Tensor._tensor_types[tensor_type] = cls

    @classmethod
    def _from_pb_tensor(cls, pb_tensor: _ir.Tensor) -> 'Tensor':
        specifc_cls = cls._tensor_types.get(pb_tensor.tensor_type(),
                                            None)  # type: ignore
        if specifc_cls is not None and cls != specifc_cls:
            return specifc_cls._from_pb_tensor(pb_tensor)

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

    def ir(self) -> 'Ir':
        from popart.ir import Ir
        return Ir._from_pb(self._pb_tensor.getIr())

    def transpose(self,
                  permutation: Optional[Iterable[int]] = None) -> 'Tensor':
        """Returns `ops.transpose(self, permutation)`."""
        import popart.ir.ops as ops
        return ops.transpose(self, permutation)

    def reshape(self, shape: Iterable[int]) -> 'Tensor':
        """Returns `ops.reshape(self, shape)`."""
        import popart.ir.ops as ops
        return ops.reshape(self, shape)

    def detach(self) -> 'Tensor':
        """Return detached tensor"""
        import popart.ir.ops as ops
        return ops.detach(self)

    def copy_to_ipu(self, destination: int,
                    source: Optional[int] = None) -> 'Tensor':
        """
        Copies a Tensor to a virtual graph.

        Args:
            destination (int):
                Ipu for the tensor to be copied to.
            source (Optional[int]):
                Ipu for the tensor to be copied from.
                By default, the source will be taken from the producer of the tensor.
                If the tensor does not have a producer a source MUST be provided.
        """
        import popart.ir.ops as ops
        return ops.ipu_copy(self, destination, source)

    @property
    def T(self) -> 'Tensor':
        """Returns the Tensor transposed with reversed axes."""
        return self.transpose()

    def __repr__(self) -> str:
        return f"Tensor[name={self.name} type={self.dtype} shape={self.shape}]"

    def __hash__(self):
        """Hashes the Tensor, based on Tensor and Ir `id`"""
        return hash((self.id, self.ir()))

    def __eq__(self, other: Any) -> bool:
        """Tensor equality, based on Tensor and Ir `id`"""
        return isinstance(
            other, Tensor) and self.id == other.id and self.ir() == other.ir()

    def __len__(self) -> int:
        """Size of 0th axis"""
        return self.shape[0]

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
        """Returns `ops.add(self, value)`."""
        import popart.ir.ops as ops
        return ops.add(self, self._ensure_tensor(value))

    def __sub__(self, value: Any) -> 'Tensor':
        """Returns `ops.sub(self, value)`."""
        import popart.ir.ops as ops
        return ops.sub(self, self._ensure_tensor(value))

    def __mul__(self, value: Any) -> 'Tensor':
        """Returns `ops.mul(self, value)`."""
        import popart.ir.ops as ops
        return ops.mul(self, self._ensure_tensor(value))

    def __truediv__(self, value: Any) -> 'Tensor':
        """Returns `ops.div(self, value)`."""
        import popart.ir.ops as ops
        return ops.div(self, self._ensure_tensor(value))

    def transpose_(self,
                   permutation: Optional[Iterable[int]] = None) -> 'Tensor':
        """Returns ops.transpose(self, permutation). Inplace"""
        import popart.ir.ops as ops
        return ops.transpose_(self, permutation)

    def reshape_(self, shape: Iterable[int]) -> 'Tensor':
        """Returns ops.reshape_(self, shape) inplace."""
        import popart.ir.ops as ops
        return ops.reshape_(self, shape)

    def flatten_(self, shape: Iterable[int]) -> 'Tensor':
        """Returns ops.flatten_(self). Inplace."""
        import popart.ir.ops as ops
        return ops.flatten_(self, shape)

    def detach_(self) -> 'Tensor':
        """Detach tensor inplace."""
        import popart.ir.ops as ops
        return ops.detach_(self)

    @property
    def T_(self) -> 'Tensor':
        """Returns the Tensor transposed with reversed axes. Inplace."""
        return self.transpose_()

    def __matmul__(self, other: Any) -> 'Tensor':
        """Returns `ops.matmul(self, other)`."""
        import popart.ir.ops as ops
        return ops.matmul(self, self._ensure_tensor(other))

    def __getitem__(self, key) -> 'Tensor':
        """
        Supports slicing and integer indexing.
        If a single index is selected during slicing the dimention will be squeezed - this matches numpy slicing rules.

        Examples:
        ```
        # Slicing
        x[0]        # Select all elements where i==0 for axis 0
        x[0,1]      # Select all elements where i==0, j==1 for axis 0 and 1
        x[0:2]      # Slice axis 0 between index 0 and 2
        x[:2,3:]    # Slice axis 0 upto 2 and axis 1 from index 3
        x[:,::-1]   # Select all elements for axis 0 and reverse axis 1

        # Integer indexing
        indices = Tensor([[0,1], [1,0]], dtype='int32')
        x[indices]  # Select elements [0,1] and [1,0] from `x`
        ```
        """

        import popart.ir.ops as ops

        if isinstance(key, (slice, int)) or (isinstance(key, tuple) and all(
                isinstance(e, (slice, int)) for e in key)):
            # Basic slicing (integer or slices)
            key = (key, ) if isinstance(key, (slice, int)) else key

            start = []
            stop = []
            step = []
            int_slices = []

            for i, key_i in enumerate(key):
                if isinstance(key_i, int):
                    start += [key_i]
                    stop += [key_i + 1]
                    step += [1]
                    int_slices += [i]

                elif isinstance(key_i, slice):
                    start += [key_i.start]
                    stop += [key_i.stop]
                    step += [key_i.step]

            out = ops.slice(self, start, stop, step)

            if len(int_slices) > 0:
                out = ops.squeeze(out, axes=int_slices)

            return out

        elif (isinstance(key, Tensor) and key.dtype.is_int):
            # Integer indexing
            return ops.gather(self, key)

        else:
            raise IndexError(
                "Only integers, slices (`:`) and integer arrays are valid indices."
            )


class Variable(Tensor, tensor_type="Variable"):
    """
    popart.ir variable tensor.
    This tensor can be used to represent a model weight or any other
    parameter that can change while running a model.
    """

    def copy_to_ipu(self, dst: int, src: int) -> 'Tensor':
        """Returns `ops.ipu_copy(self, dst, src)`.
            Must provide a src value."""
        import popart.ir.ops as ops
        return ops.ipu_copy(self, dst, src)


class Constant(Tensor, tensor_type="Const"):
    """
    popart.ir constant tensor.
    This tensor cannot change during the runtime of a model.
    """

    def copy_to_ipu(self, dst: int, src: int) -> 'Tensor':
        """Returns ops.ipu_copy(self, dst, src).
            Must provide a src value."""
        import popart.ir.ops as ops
        return ops.ipu_copy(self, dst, src)


downcast_np_dtypes = {
    np.dtype('int64'): np.dtype('int32'),
    np.dtype('uint64'): np.dtype('uint32'),
    np.dtype('float64'): np.dtype('float32'),
}


def variable(
        data: Union[np.ndarray, Iterable[Any], int, float, bool],
        dtype: Optional[dtypes.dtype] = None,
        name: Optional[str] = None,
        downcast: bool = True,
) -> Variable:
    """
    A variable tensor that is initialised with data during graph creation.

    This tensor can be used to represent a model weight or any other
    parameter that can change while running a model.

    Must be created in the main graph scope. Example:
    ```
    import popart.ir as pir
    with pir.Ir().main_graph():
        a = pir.variable(0)
    ```

    Args:
        data (np.ndarray, or a value numpy can use to construct an np.ndarray):
            The data used to initialise the tensor.
        dtype (Optional[dtype]):
            The data type of the tensor to be created, if not specified Numpy will infer the data
            type and be downcasted to 32 bits if necessary.
        name (Optional[str]):
            The name of the tensor. Defaults to `None`.
        downcast (bool):
            If no dtype is provided 64 bit float/ints will be downcasted to 32 bit variants. Default to True.
    """
    g = gcg()
    pb_g = g._pb_graph
    np_dtype = dtype.as_numpy() if dtype is not None else None
    np_data: np.ndarray = np.array(data, dtype=np_dtype)
    if np_data.dtype in downcast_np_dtypes and downcast and dtype is None:
        np_data = np_data.astype(downcast_np_dtypes[np_data.dtype])
    pir_dt = dtypes.dtype.as_dtype(np_data)
    info = _ir.TensorInfo(pir_dt._pb_dtype, np_data.shape)
    pb_id = g._create_tensor_id(name)
    pb_g.addVarInit(pb_id, info, np_data)
    return Variable._from_pb_tensor(pb_g.getTensor(pb_id))


def constant(
        data: Union[np.ndarray, Iterable[Any], int, float],
        dtype: Optional[dtypes.dtype] = None,
        name: Optional[str] = None,
        downcast: bool = True,
) -> Constant:
    """A constant tensor that is initialised with data during graph creation.

    This tensor cannot change during the runtime of a model. The intended use
    of this class is when doing operations between `popart.ir.Tensor`
    instances and other types, such as `numpy.ndarray` objects, numbers, or
    list or tuples of numbers.

    Example:
    ```
    import popart.ir as pir
    ir = pir.Ir()
    with ir.main_graph():
        a = pir.constant(0)
        # The `1` will be implicitly converted to a `Constant`.
        b = a + 1
    ```

    Args:
        data (np.array, or a value numpy can use to construct an np.ndarray):
            The data used to initialise the tensor.
        dtype (Optional[dtype]):
            The data type of the tensor to be created, if not specified Numpy will infer the data
            type and be downcasted to 32 bits if necessary.
        name (Optional[str]):
            The name of the tensor. Defaults to `None`.
    """
    g = gcg()
    pb_g = g._pb_graph
    np_dtype = dtype.as_numpy() if dtype is not None else None
    np_data: np.ndarray = np.array(data, dtype=np_dtype)
    if np_data.dtype in downcast_np_dtypes and downcast and dtype is None:
        np_data = np_data.astype(downcast_np_dtypes[np_data.dtype])
    pir_dt = dtypes.dtype.as_dtype(np_data)
    info = _ir.TensorInfo(pir_dt._pb_dtype, np_data.shape)
    pb_id = g._create_tensor_id(name)
    pb_g.addConstInit(pb_id, info, np_data)
    return Constant._from_pb_tensor(pb_g.getTensor(pb_id))


def subgraph_input(shape: Iterable[int],
                   dtype: dtypes.dtype,
                   name: Optional[str] = None) -> Tensor:
    """Create a new input tensor to the current graph.

    You can use this function when defining a subgraph to create a new input
    tensor. When you call that subgraph, you will have to pass a tensor to the
    subgraph for this input.

    Example:
    ```
    import popart.ir as pir
    
    def add_w(x):
        w = pir.subgraph_input(x.shape, x.dtype, "w")
        return w + x
    
    ir = pir.Ir()
    with ir.main_graph():
        w = pir.variable(1)
        x = pir.variable(3)
        add_w_graph = ir.create_graph(add_w, x, w)
        y = ops.call(add_w_graph, x, w)
    ```

    Args:
        shape (Tuple[int, ...])
            The shape of the Tensor
        dtype (dtype):
            The data type of the tensor
        name (Optional[str]):
            The name of the tensor.
    """
    g = gcg()
    pb_g = g._pb_graph

    pb_id = g._create_tensor_id(name)
    pb_info = _ir.TensorInfo(dtype._pb_dtype, list(shape))

    pb_g.addInput(pb_id, pb_info)

    return Tensor._from_pb_tensor(pb_g.getTensor(pb_id))


def subgraph_output(t: Tensor) -> None:
    """Mark a tensor as an output in the current graph.

    You can use this function when defining a subgraph to mark an existing
    tensor in the subgraph as an output. When you call that subgraph, it will
    return that tensor in the parent graph.

    Example:
    ```
    import popart.ir as pir
    
    def add_w(x):
        w = pir.subgraph_input(x.shape, x.dtype, "w")
        y = w + x
        pir.subgraph_output(y)
    
    ir = pir.Ir()
    with ir.main_graph():
        w = pir.variable(1)
        x = pir.variable(3)
        add_w_graph = ir.create_graph(add_w, x, w)
        y = ops.call(add_w_graph, x, w)
    ```

    Args:
        t (Tensor):
            The subgraph tensor to mark as an output in the current graph.

    Throws:
        ValueError:
            If #t is not in the current graph.
    """
    g = gcg()
    pb_g = g._pb_graph

    from popart.ir.ops.utils import check_in_graph

    check_in_graph(g, t)

    pb_g.markAsOutput(t.id)
