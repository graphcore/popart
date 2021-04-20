// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_HOSTX_HPP
#define GUARD_NEURALNET_HOSTX_HPP

#include <popart/popx/opx.hpp>

namespace popart {

namespace popx {

class HostBaseOpx : public Opx {
public:
  HostBaseOpx(Op *, Devicex *);

protected:
  poplar::Tensor
  load(poplar::program::Sequence &, const TensorId &, const TensorId &) const;
  void
  store(poplar::program::Sequence &, const TensorId &, const TensorId &) const;
};

class HostLoadOpx : public HostBaseOpx {
public:
  HostLoadOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

  InputCreatorType getInputCreatorType(InIndex) const final;
  poplar::Tensor
      unwindTensorLayout(poplar::Tensor, InIndex, OutIndex) const final;
  view::RegMap unwindRegion(InIndex, OutIndex) const final;
};

class HostStoreOpx : public HostBaseOpx {
public:
  HostStoreOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

  InputCreatorType getInputCreatorType(InIndex) const final;
  poplar::Tensor
      unwindTensorLayout(poplar::Tensor, InIndex, OutIndex) const final;
  view::RegMap unwindRegion(InIndex, OutIndex) const final;
};

} // namespace popx
} // namespace popart

#endif
