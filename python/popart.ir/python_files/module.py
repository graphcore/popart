# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from typing import TYPE_CHECKING, Union, Iterable

if TYPE_CHECKING:
    from popart.ir import Tensor


class Module:
    """
  Callable class from which user-defined layers can inherit.

  The #build method should be overriden and should build the subgraph.

  The benefit of inheriting from this class rather than passing a function is
  that you can save input tensors as fields on `self`, then later when you call
  the subgraph, you can pass a mapping from the input tensor ids to the
  corresponding parent tensor you wish to pass.
  """

    def __call__(self, *args,
                 **kwargs) -> Union[None, 'Tensor', Iterable['Tensor']]:
        return self.build(*args, **kwargs)

    def build(self, *args,
              **kwargs) -> Union[None, 'Tensor', Iterable['Tensor']]:
        raise NotImplementedError(
            "Your popart.ir.Module must override `build` method")
