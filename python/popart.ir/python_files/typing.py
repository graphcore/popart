# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import TypeVar

T = TypeVar('T')


def NewAliasAnnotation(name: str, tp: T) -> T:
    """Create an alias of `tp` that will be interpreted by static type checkers
        as `tp` but at runtime can be identified as a unique symbol. For example:
        ```
        Id = NewAliasAnnotation('Id', int)
        def fn(x: Id):
            return x + 1
        fn(1)  # successful type check

        inspect.signature(fn).parameters['x'].annotation is Id  # True
        ```

    Args:
        name (str): Name to provide the alias annotation
        tp (T): Type to alias
    """

    def symbol():
        raise NotImplementedError(
            f"{name} should only be used as an annotation. "
            f"Try using the associated type alias instead: {tp}")

    symbol.__name__ = name
    return symbol  # type: ignore
