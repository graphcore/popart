// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_TIEDGATHERX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_TIEDGATHERX_HPP_

#include <poplar/Tensor.hpp>
#include <popart/popx/op/gatherx.hpp>

#include "popart/names.hpp"
#include "popart/popx/debugcontextx.hpp"
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

class TiedGatherOpx final : public GatherBaseOpx {
public:
  TiedGatherOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

  InputCreatorType getInputCreatorType(int index0) const final;

  poplar::Tensor createInput(InIndex index,
                             const poplar::DebugNameAndId &dnai) const final;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_TIEDGATHERX_HPP_
