// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_REPLICATEDREDUCESCATTERX_HPP
#define GUARD_NEURALNET_REPLICATEDREDUCESCATTERX_HPP

#include <popart/popx/op/collectives/collectivesx.hpp>

namespace popart {
namespace popx {

class ReplicatedReduceScatterOpx : public CollectivesBaseOpx {
public:
  ReplicatedReduceScatterOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

  poplar::Tensor createInput(InIndex,
                             const poplar::DebugNameAndId &dnai) const final;
  InputCreatorType getInputCreatorType(InIndex) const final;
  std::vector<TensorId> mustExistBeforeCreate(InIndex index0) const final;
  bool hasCreatorViewChangers(InIndex index) const final;
  ViewChangers getCreatorViewChangers(InIndex index) const final;
};

} // namespace popx
} // namespace popart

#endif
