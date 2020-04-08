// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_COLLECTIVEX_HPP
#define GUARD_NEURALNET_COLLECTIVEX_HPP

#include <popart/names.hpp>
#include <popart/popx/opx.hpp>

namespace popart {

namespace popx {

class ReplicatedAllReduceInplaceOpx : public Opx {
public:
  ReplicatedAllReduceInplaceOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

class ReplicatedAllReduceOpx : public Opx {
public:
  ReplicatedAllReduceOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

class ReplicatedReduceScatterOpx : public Opx {
public:
  ReplicatedReduceScatterOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
