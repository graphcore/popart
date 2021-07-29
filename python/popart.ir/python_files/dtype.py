# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
"""Definition and instances of a class to represent `Tensor` data types."""


class dtype:
    def __init__(self) -> None:
        """A class to represent the type of elements in a tensor.

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
            TypeError: This class cannot be initialised.
        """
        self._is_complex: bool = None
        self._is_floating_point: bool = None
        self._is_signed: bool = None
        self._name: str = None
        raise TypeError(f"Cannot create {self.__module__}.dtype instances.")

    @property
    def is_complex(self) -> bool:
        return self._is_complex

    @property
    def is_floating_point(self) -> bool:
        return self._is_floating_point

    @property
    def is_signed(self) -> bool:
        return self._is_signed

    def __repr__(self) -> str:
        return f'{self.__module__}.{self._name}'

    @classmethod
    def _factory(
            cls,
            name: str,
            is_complex: bool,
            is_floating_point: bool,
            is_signed: bool,
    ) -> 'dtype':
        """Factory method to construct `dtype` instances.

        Args:
            name (str):
                The name of the `dtype`. Used in `__repr__()`.
            is_complex (bool):
                Is the type complex.
            is_floating_point (bool):
                Is the type floating point.
            is_signed (bool):
                Is the type signed.

        Returns:
            dtype:
                A new `dtype` instance.
        """
        self: 'dtype' = super().__new__(cls)
        self._is_complex = is_complex
        self._is_floating_point = is_floating_point
        self._is_signed = is_signed
        self._name = name
        return self


# Fixed point types
bool = dtype._factory('bool', False, False, False)
int8 = dtype._factory('int8', False, False, True)
int16 = dtype._factory('int16', False, False, True)
int32 = dtype._factory('int32', False, False, True)
int64 = dtype._factory('int64', False, False, True)
uint8 = dtype._factory('uint8', False, False, False)
uint16 = dtype._factory('uint16', False, False, False)
uint32 = dtype._factory('uint32', False, False, False)
uint64 = dtype._factory('uint64', False, False, False)

# Floating point types
float16 = dtype._factory('float16', False, True, True)
float32 = dtype._factory('float32', False, True, True)
float64 = dtype._factory('float64', False, True, True)

# Complex types
complex64 = dtype._factory('complex64', True, False, True)
complex128 = dtype._factory('complex128', True, False, True)

# Type aliases
half = float16
float = float32
double = float64

# Delete the `dtype` factory from the `dtype` class.
del dtype._factory

# A set of objects that won't be imported when using `from dtype import *`.
exclude_from_all = set([name for name in dir() if name.startswith('_')])
exclude_from_all.add('exclude_from_all')

__all__ = [name for name in dir() if name not in exclude_from_all]
