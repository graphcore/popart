// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_REMOTEX_HPP
#define GUARD_NEURALNET_REMOTEX_HPP

#include <popart/popx/opx.hpp>

namespace popart {

namespace popx {

class RemoteStoreOpx : public Opx {
public:
  RemoteStoreOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

class RemoteLoadOpx : public Opx {
public:
  RemoteLoadOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
  InputCreatorType getInputCreatorType(InIndex index) const final;
  poplar::Tensor
      unwindTensorLayout(poplar::Tensor, InIndex, OutIndex) const final;
  view::RegMap unwindRegion(InIndex, OutIndex) const final;
};

} // namespace popx
} // namespace popart

#endif
