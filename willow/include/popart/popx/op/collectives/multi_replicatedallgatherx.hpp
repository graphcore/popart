// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_MULTIREPLICATEDALLGATHERX_HPP
#define GUARD_NEURALNET_MULTIREPLICATEDALLGATHERX_HPP
#include <snap/Tensor.hpp>
#include <popart/popx/op/collectives/collectivesx.hpp>

#include "popart/names.hpp"
#include "popart/popx/popopx.hpp"

namespace snap {
namespace program {
class Sequence;
} // namespace program
} // namespace snap

namespace popart {
class Op;

namespace popx {
class Devicex;

/**
 * Lowers the MultiReplicatedAllGatherOp to Poplar by growing each
 * individual output tensor, and performing a to-destination all-gather
 * on a concatenation of the input tensors. CBR features and logic
 * are transplanted from the ReplicatedAllGatherOpx implementation
 */
class MultiReplicatedAllGatherOpx : public MultiCollectiveBaseOpx {
public:
  MultiReplicatedAllGatherOpx(popart::Op *op, Devicex *devicex);
  InputCreatorType getInputCreatorType(InIndex) const override;
  snap::Tensor unwindTensorLayout(snap::Tensor tensor,
                                  InIndex in,
                                  OutIndex out) const override;
  view::RegMap unwindRegion(InIndex, OutIndex) const override;
  void growPart(OpxGrowPartId id) const override;
  void grow(snap::program::Sequence &prog) const override;
  ViewChangers getCreatorViewChangers(InIndex index) const override;
  snap::Tensor
  createInputTensor(InIndex idx,
                    const poplar::DebugNameAndId &dnai) const override;
  bool hasCreatorViewChangers(InIndex index) const override;
  std::set<TensorId> mustExistBeforeCreate(InIndex) const override;
};

} // namespace popx
} // namespace popart

#endif
