// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_ADADELTAUPDATERX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_ADADELTAUPDATERX_HPP_

#include "popart/popx/debugcontextx.hpp"
#include <set>
#include <poplar/Tensor.hpp>
#include <popart/popx/opx.hpp>

#include "popart/names.hpp"

namespace poplar {
namespace program {
class Sequence;
} // namespace program
} // namespace poplar

namespace popart {
class Op;

namespace popx {
class Devicex;
class ViewChangers;

class AdaDeltaUpdaterOpx : public Opx {
public:
  AdaDeltaUpdaterOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

  // can create the accumulator2 input Tensor (@Accl2 index)
  // from the weight gradient tensor (@Accl1 index)
  poplar::Tensor createInput(InIndex,
                             const poplar::DebugNameAndId &dnai) const final;
  InputCreatorType getInputCreatorType(InIndex) const final;
  std::set<TensorId> mustExistBeforeCreate(InIndex) const final;
  bool hasCreatorViewChangers(InIndex index) const final;
  ViewChangers getCreatorViewChangers(InIndex index) const final;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_ADADELTAUPDATERX_HPP_
