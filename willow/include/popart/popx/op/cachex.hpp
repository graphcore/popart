// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_CACHEX_HPP
#define GUARD_NEURALNET_CACHEX_HPP

#include <popart/popx/opx.hpp>

namespace popart {

namespace popx {

class CacheStoreOpx : public Opx {
public:
  CacheStoreOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

class CacheLoadOpx : public Opx {
public:
  CacheLoadOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
  InputCreatorType getInputCreatorType(InIndex index) const final;
  poplar::Tensor
      unwindTensorLayout(poplar::Tensor, InIndex, OutIndex) const final;
  view::RegMap unwindRegion(InIndex, OutIndex) const final;
};

} // namespace popx
} // namespace popart

#endif
