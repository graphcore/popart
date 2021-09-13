// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_REMOTEX_HPP
#define GUARD_NEURALNET_REMOTEX_HPP

#include <popart/popx/op/exchange/exchangex.hpp>
#include <popart/popx/popopx.hpp>

namespace popart {

namespace popx {

class RemoteBaseOpx : public ExchangeBaseOpx {
public:
  RemoteBaseOpx(Op *, Devicex *);
};

class RemoteStoreOpx : public RemoteBaseOpx {
public:
  RemoteStoreOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
};

class RemoteLoadOpx : public RemoteBaseOpx {
public:
  RemoteLoadOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
  InputCreatorType getInputCreatorType(InIndex index) const final;
  snap::Tensor unwindTensorLayout(snap::Tensor, InIndex, OutIndex) const final;
  view::RegMap unwindRegion(InIndex, OutIndex) const final;
};

} // namespace popx
} // namespace popart

#endif
