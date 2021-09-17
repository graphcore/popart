// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_REPLICATEDALLREDUCEX_HPP
#define GUARD_NEURALNET_REPLICATEDALLREDUCEX_HPP

#include <popart/popx/op/collectives/collectivesx.hpp>

namespace popart {
namespace popx {

class ReplicatedAllReduceOpx : public CollectivesBaseOpx {
public:
  ReplicatedAllReduceOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const override;
  InputCreatorType getInputCreatorType(InIndex index) const final;
  snap::Tensor unwindTensorLayout(snap::Tensor, InIndex, OutIndex) const final;
  view::RegMap unwindRegion(InIndex, OutIndex) const final;
};

class ReplicatedAllReduceInplaceOpx : public ReplicatedAllReduceOpx {
public:
  ReplicatedAllReduceInplaceOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
