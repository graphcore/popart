// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_COLLECTIVES_REPLICATEDREDUCESCATTERX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_COLLECTIVES_REPLICATEDREDUCESCATTERX_HPP_

#include <poplar/Tensor.hpp>
#include <popart/popx/op/collectives/collectivesx.hpp>

#include "popart/names.hpp"
#include "popart/popx/debugcontextx.hpp"
#include "popart/popx/opx.hpp"
#include "popart/popx/viewchangers.hpp"

namespace poplar {
namespace program {
class Sequence;
} // namespace program
} // namespace poplar

namespace popart {
class Op;

namespace popx {
class Devicex;

class ReplicatedReduceScatterOpx : public CollectivesBaseOpx {
public:
  ReplicatedReduceScatterOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

  poplar::Tensor createInput(InIndex,
                             const poplar::DebugNameAndId &dnai) const final;
  InputCreatorType getInputCreatorType(InIndex) const final;
  DnfTensorIds mustExistBeforeCreateDNF(InIndex index0) const final;
  bool hasCreatorViewChangers(InIndex index) const final;
  ViewChangers getCreatorViewChangers(InIndex index) const final;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_COLLECTIVES_REPLICATEDREDUCESCATTERX_HPP_
