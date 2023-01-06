// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_MULTIREPLICATEDALLGATHERX_HPP
#define GUARD_NEURALNET_MULTIREPLICATEDALLGATHERX_HPP
#include <set>
#include <poplar/Tensor.hpp>
#include <popart/popx/op/collectives/collectivesx.hpp>

#include "popart/names.hpp"
#include "popart/popx/debugcontextx.hpp"
#include "popart/popx/opx.hpp"
#include "popart/popx/viewchangers.hpp"
#include "popart/tensordebuginfo.hpp"

namespace poplar {
namespace program {
class Sequence;
} // namespace program
} // namespace poplar

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
  poplar::Tensor unwindTensorLayout(poplar::Tensor tensor,
                                    InIndex in,
                                    OutIndex out) const override;
  view::RegMap unwindRegion(InIndex, OutIndex) const override;
  void growPart(OpxGrowPartId id) const override;
  void grow(poplar::program::Sequence &prog) const override;
  ViewChangers getCreatorViewChangers(InIndex index) const override;
  poplar::Tensor createInput(InIndex idx,
                             const poplar::DebugNameAndId &dnai) const override;
  bool hasCreatorViewChangers(InIndex index) const override;
  std::set<TensorId> mustExistBeforeCreate(InIndex) const override;
};

} // namespace popx
} // namespace popart

#endif
