// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_REPLICATEDALLGATHERX_HPP
#define GUARD_NEURALNET_REPLICATEDALLGATHERX_HPP

#include <popart/popx/op/collectives/collectivesx.hpp>

namespace popart {
namespace popx {

class ReplicatedAllGatherOpx : public CollectivesBaseOpx {
public:
  ReplicatedAllGatherOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
  InputCreatorType getInputCreatorType(InIndex index) const final;
  poplar::Tensor
      unwindTensorLayout(poplar::Tensor, InIndex, OutIndex) const final;
  view::RegMap unwindRegion(InIndex, OutIndex) const final;
  poplar::Tensor createInput(InIndex index,
                             const std::string &name) const final;
  std::vector<TensorId> mustExistBeforeCreate(InIndex index0) const final;
  bool hasCreatorViewChangers(InIndex index) const final;
  ViewChangers getCreatorViewChangers(InIndex index) const final;
};

} // namespace popx
} // namespace popart

#endif
