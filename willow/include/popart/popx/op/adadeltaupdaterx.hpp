// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_ADADELTAUPDATERX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_ADADELTAUPDATERX_HPP_

#include "popart/popx/debugcontextx.hpp"
#include <set>
#include <snap/Tensor.hpp>
#include <popart/popx/popopx.hpp>

#include "popart/names.hpp"

namespace snap {
namespace program {
class Sequence;
} // namespace program
} // namespace snap

namespace popart {
class Op;

namespace popx {
class Devicex;
class ViewChangers;

class AdaDeltaUpdaterOpx : public PopOpx {
public:
  AdaDeltaUpdaterOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;

  // can create the accumulator2 input Tensor (@Accl2 index)
  // from the weight gradient tensor (@Accl1 index)
  snap::Tensor
  createInputTensor(InIndex, const poplar::DebugNameAndId &dnai) const final;
  InputCreatorType getInputCreatorType(InIndex) const final;
  std::set<TensorId> mustExistBeforeCreate(InIndex) const final;
  bool hasCreatorViewChangers(InIndex index) const final;
  ViewChangers getCreatorViewChangers(InIndex index) const final;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_ADADELTAUPDATERX_HPP_
