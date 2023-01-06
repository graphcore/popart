// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_COLLECTIVES_REPLICATEDALLGATHERX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_COLLECTIVES_REPLICATEDALLGATHERX_HPP_

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

class ReplicatedAllGatherOpx : public CollectivesBaseOpx {
public:
  ReplicatedAllGatherOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
  InputCreatorType getInputCreatorType(InIndex index) const final;
  poplar::Tensor
      unwindTensorLayout(poplar::Tensor, InIndex, OutIndex) const final;
  view::RegMap unwindRegion(InIndex, OutIndex) const final;
  poplar::Tensor createInput(InIndex index,
                             const poplar::DebugNameAndId &dnai) const final;
  std::set<TensorId> mustExistBeforeCreate(InIndex index0) const final;
  bool hasCreatorViewChangers(InIndex index) const final;
  ViewChangers getCreatorViewChangers(InIndex index) const final;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_COLLECTIVES_REPLICATEDALLGATHERX_HPP_
