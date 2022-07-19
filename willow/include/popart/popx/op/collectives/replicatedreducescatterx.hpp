// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_COLLECTIVES_REPLICATEDREDUCESCATTERX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_COLLECTIVES_REPLICATEDREDUCESCATTERX_HPP_

#include "popart/popx/debugcontextx.hpp"
#include <snap/Tensor.hpp>
#include <popart/popx/op/collectives/collectivesx.hpp>

#include "popart/names.hpp"
#include "popart/popx/popopx.hpp"
#include "popart/popx/viewchangers.hpp"

namespace snap {
namespace program {
class Sequence;
} // namespace program
} // namespace snap

namespace popart {
class Op;

namespace popx {
class Devicex;

class ReplicatedReduceScatterOpx : public CollectivesBaseOpx {
public:
  ReplicatedReduceScatterOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;

  snap::Tensor
  createInputTensor(InIndex, const poplar::DebugNameAndId &dnai) const final;
  InputCreatorType getInputCreatorType(InIndex) const final;
  DnfTensorIds mustExistBeforeCreateDNF(InIndex index0) const final;
  bool hasCreatorViewChangers(InIndex index) const final;
  ViewChangers getCreatorViewChangers(InIndex index) const final;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_COLLECTIVES_REPLICATEDREDUCESCATTERX_HPP_
