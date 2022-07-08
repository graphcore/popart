# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
"""Definition and instances of a class to represent `Tensor` data types."""

import builtins
from typing import Any, Mapping
import numpy as np
import popart._internal.ir as _ir

try:
    import torch
except ModuleNotFoundError:
    torch = None

# A dictionary to map from numpy to popxl types.
_NP_TO_POPXL: Mapping[np.dtype, "dtype"] = {}
# A dictionary to map from string to popxl types.
_STR_TO_POPXL: Mapping[str, "dtype"] = {}
# A dictionary to map from Python to popxl types.
_PY_TO_POPXL: Mapping[Any, "dtype"] = {}
# A dictionart to map from popart._ir to popxl types.
_PB_TO_POPXL: Mapping[_ir.DataType, "dtype"] = {}
# A dictionary to map from PyTorch to popxl types
_PT_TO_POPXL: Mapping[Any, "dtype"] = {}


class dtype:
    def __init__(self) -> None:
        """
        Construct the dtype.

        A class to represent the type of elements in a tensor.

        Available data types are:

            - `bool`
            - `int8`
            - `int16`
            - `int32`
            - `int64`
            - `uint8`
            - `uint16`
            - `uint32`
            - `uint64`
            - `float16`
            - `float32`
            - `float64`
            - `complex64`
            - `complex128`

        Some data types have aliases:

            - `half` aliases `float16`
            - `float` aliases `float32`
            - `double` aliases `float64`

        Raises:
            TypeError: If an attempt is made to create an object of this class.
        """
        self._is_complex: bool = None
        self._is_int: bool = None
        self._is_floating_point: bool = None
        self._is_signed: bool = None
        self._name: str = None
        self._np_type: np.dtype = None
        self._pt_type = None
        self._py_type = None
        self._pb_dtype: _ir.DataType = None
        raise TypeError(f"Cannot create {self.__module__}.dtype instances.")

    @property
    def name(self) -> str:
        """Return the name of dtype."""
        return self._name

    @property
    def is_complex(self) -> bool:
        """Return True if a complex number dtype. False otherwise."""
        return self._is_complex

    @property
    def is_int(self) -> bool:
        """Return True if an integer number dtype. False otherwise."""
        return self._is_int

    @property
    def is_floating_point(self) -> bool:
        """Return True if a float point number dtype. False otherwise."""
        return self._is_floating_point

    @property
    def is_signed(self) -> bool:
        """Return True if a signed number dtype. False otherwise."""
        return self._is_signed

    @classmethod
    def from_name(cls, name: str) -> "dtype":
        """Convert a dtype name into a dtype.

        Args:
            name (str): Dtype name

        Raises:
            ValueError: If name does not exist

        Returns:
            dtype
        """
        try:
            return _STR_TO_POPXL[name]
        except (KeyError, TypeError):
            pass

        raise ValueError(f"There is not a `popxl.dtype` that has a name: {name}.")

    @classmethod
    def as_dtype(cls, type_value: Any) -> "dtype":
        """Convert the given `type_value` to a `popxl.dtype`.

        Args:
            type_value:
                A `numpy.dtype`, `torch.dtype`, string, a built-in Python type or a
                `numpy.ndarray`, `torch.tensor` which can be converted to a `popxl.dtype`.

        Raises:
            ValueError:
                If `type_value` cannot be converted to a `popxl.dtype`.

        Returns:
            dtype: A `popxl.dtype` corresponding to `type_value`.
        """
        try:
            return _STR_TO_POPXL[type_value]
        except (KeyError, TypeError):
            pass

        try:
            return _PY_TO_POPXL[type_value]
        except (KeyError, TypeError):
            pass

        try:
            return _PB_TO_POPXL[type_value]
        except (KeyError, TypeError):
            pass

        try:
            return _NP_TO_POPXL[type_value]
        except (KeyError, TypeError):
            pass

        try:
            return _PT_TO_POPXL[type_value]
        except (KeyError, TypeError):
            pass

        try:
            return _NP_TO_POPXL[np.dtype(type_value)]
        except (KeyError, TypeError):
            pass

        if hasattr(type_value, "dtype"):
            try:
                return _NP_TO_POPXL[type_value.dtype]
            except (KeyError, TypeError):
                pass

            try:
                return _PT_TO_POPXL[type_value.dtype]
            except (KeyError, TypeError):
                pass

        raise ValueError(
            f"There is not a `popxl.dtype` that is compatible"
            f" with value: {type_value}, type: {type(type_value)}."
        )

    def as_numpy(self) -> np.dtype:
        """Convert the `popxl.dtype` to a corresponding `numpy.dtype`.

        Raises:
            TypeError:
                If the `popxl.dtype` cannot be converted to a `numpy.dtype`.

        Returns:
            np.dtype:
                A `numpy.dtype` corresponding to `popxl.dtype`.
        """
        if self._np_type is not None:
            return self._np_type
        else:
            raise TypeError(
                f"`popxl.{self._name}` does not have a " f"corresponding NumPy dtype."
            )

    def as_torch(self) -> np.dtype:
        """Convert the `popxl.dtype` to a corresponding `torch.dtype`.

        Raises:
            TypeError:
                If the `popxl.dtype` cannot be converted to a `torch.dtype`.
            ModuleNotFoundError:
                If PyTorch is not installed.

        Returns:
            torch.dtype:
                A `torch.dtype` corresponding to `popxl.dtype`.
        """
        if torch is None:
            raise ModuleNotFoundError("PyTorch is not installed.")
        if self._pt_type is not None:
            return self._pt_type
        else:
            raise TypeError(
                f"`popxl.{self._name}` does not have a " f"corresponding PyTorch dtype."
            )

    def __repr__(self) -> str:
        return f"{self.__module__}.{self._name}"

    @classmethod
    def _factory(
        cls,
        name: str,
        is_complex: bool,
        is_int: bool,
        is_floating_point: bool,
        is_signed: bool,
        np_type: np.dtype,
        pt_type: str,
        py_type: Any,
        pb_type: _ir.DataType,
    ) -> "dtype":
        """Construct `dtype` instances.

        Used as a factory method.

        Args:
            name (str):
                The name of the `dtype`. Used in `__repr__()`.
            is_complex (bool):
                Is the type complex.
            is_int (bool):
                Is the type an integer (signed or unsigned).
            is_floating_point (bool):
                Is the type floating point.
            is_signed (bool):
                Is the type signed.
            np_type (np.dtype):
                A `numpy.dtype` that corresponds to the created `dtype`.
            pt_type (str):
                A PyTorch dtype name that corresponds to the created `dtype`.
            py_type (Any):
                Python native type that corresponds to the created `dtype`.
            pb_type (_ir.DataType):
                The corresponding low-level `pybind11` `DataType`.

        Returns:
            dtype:
                A new `dtype` instance.
        """
        global _NP_TO_POPXL, _STR_TO_POPXL, _PY_TO_POPXL, _PT_TO_POPXL

        self: "dtype" = super().__new__(cls)
        self._is_complex = is_complex
        self._is_int = is_int
        self._is_floating_point = is_floating_point
        self._is_signed = is_signed
        self._name = name

        if np_type is not None:
            assert np_type not in _NP_TO_POPXL
            np_type = np.dtype(np_type)
            _NP_TO_POPXL[np_type] = self
            self._np_type = np_type

        if pt_type is not None and torch is not None:
            assert pt_type not in _PT_TO_POPXL
            pt_type = getattr(torch, pt_type)
            _PT_TO_POPXL[pt_type] = self
            self._pt_type = pt_type

        assert name not in _STR_TO_POPXL
        _STR_TO_POPXL[name] = self

        if py_type is not None:
            assert py_type not in _PY_TO_POPXL
            _PY_TO_POPXL[py_type] = self
            self._py_type = py_type

        if pb_type is not None:
            assert py_type not in _PB_TO_POPXL
            _PB_TO_POPXL[pb_type] = self
            self._pb_dtype = pb_type

        return self


# yapf: disable
# Fixed point types
# Args: name, is_complex, is_int, is_floating_point, is_signed, np_type, pt_type, py_type, pb_type
bool = dtype._factory('bool', False, False, False, False, 'bool_', 'bool', builtins.bool, _ir.DataType.BOOL)
int8 = dtype._factory('int8', False, True, False, True, 'int8', 'int8', None, _ir.DataType.INT8)
int16 = dtype._factory('int16', False, True, False, True, 'int16', 'int16', None, _ir.DataType.INT16)
int32 = dtype._factory('int32', False, True, False, True, 'int32', 'int32', None, _ir.DataType.INT32)
int64 = dtype._factory('int64', False, True, False, True, 'int64', 'int64', None, _ir.DataType.INT64)
uint8 = dtype._factory('uint8', False, True, False, False, 'uint8', 'uint8', None, _ir.DataType.UINT8)
uint16 = dtype._factory('uint16', False, True, False, False, 'uint16', None, None, _ir.DataType.UINT16)
uint32 = dtype._factory('uint32', False, True, False, False, 'uint32', None, None, _ir.DataType.UINT32)
uint64 = dtype._factory('uint64', False, True, False, False, 'uint64', None, None, _ir.DataType.UINT64)

# Floating point types
float16 = dtype._factory('float16', False, False, True, True, 'float16', 'float16', None, _ir.DataType.FLOAT16)
float32 = dtype._factory('float32', False, False, True, True, 'float32', 'float32', builtins.float, _ir.DataType.FLOAT)
float64 = dtype._factory('float64', False, False, True, True, 'float64', 'float64', None, _ir.DataType.DOUBLE)

# Complex types
complex64 = dtype._factory('complex64', True, False, False, True, 'complex64', 'complex64', None, _ir.DataType.COMPLEX64)
complex128 = dtype._factory('complex128', True, False, False, True, 'complex128', 'complex128', None, _ir.DataType.COMPLEX128)
# yapf: enable

# Type aliases
half = float16
float = float32
double = float64

# Delete the `dtype` factory from the `dtype` class.
del dtype._factory
