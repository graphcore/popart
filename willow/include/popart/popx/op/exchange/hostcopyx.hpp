// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_HOSTX_HPP
#define GUARD_NEURALNET_HOSTX_HPP

#include <popart/popx/op/exchange/exchangex.hpp>
#include <popart/popx/popopx.hpp>

namespace popart {

namespace popx {

class HostBaseOpx : public ExchangeBaseOpx {
public:
  HostBaseOpx(Op *, Devicex *);
};

class HostLoadOpx : public HostBaseOpx {
public:
  HostLoadOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;

  InputCreatorType getInputCreatorType(InIndex) const final;
  snap::Tensor
  createInputTensor(InIndex index,
                    const poplar::DebugNameAndId &dnai) const final;
  snap::Tensor unwindTensorLayout(snap::Tensor, InIndex, OutIndex) const final;
  view::RegMap unwindRegion(InIndex, OutIndex) const final;
};

class HostLoadInplaceOpx : public HostLoadOpx {
public:
  HostLoadInplaceOpx(Op *, Devicex *);
};

class HostStoreOpx : public HostBaseOpx {
public:
  HostStoreOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
  snap::Tensor
  createInputTensor(InIndex index,
                    const poplar::DebugNameAndId &dnai) const final;
  InputCreatorType getInputCreatorType(InIndex) const final;
};

} // namespace popx
} // namespace popart

#endif
