// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_COLLECTIVES_REPLICATEDALLREDUCEX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_COLLECTIVES_REPLICATEDALLREDUCEX_HPP_

#include <poplar/Tensor.hpp>
#include <popart/popx/op/collectives/collectivesx.hpp>

#include "popart/names.hpp"
#include "popart/popx/opx.hpp"

namespace poplar {
namespace program {
class Sequence;
} // namespace program
} // namespace poplar

namespace popart {
class Op;

namespace popx {
class Devicex;

class ReplicatedAllReduceOpx : public CollectivesBaseOpx {
public:
  ReplicatedAllReduceOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const override;
  InputCreatorType getInputCreatorType(InIndex index) const final;
  poplar::Tensor
      unwindTensorLayout(poplar::Tensor, InIndex, OutIndex) const final;
  view::RegMap unwindRegion(InIndex, OutIndex) const final;
};

class ReplicatedAllReduceInplaceOpx : public ReplicatedAllReduceOpx {
public:
  ReplicatedAllReduceInplaceOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_COLLECTIVES_REPLICATEDALLREDUCEX_HPP_
