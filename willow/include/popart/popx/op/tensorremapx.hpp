// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_TENSORREMAPX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_TENSORREMAPX_HPP_

#include <snap/Tensor.hpp>
#include <popart/names.hpp>
#include <popart/popx/popopx.hpp>

namespace snap {
namespace program {
class Sequence;
} // namespace program
} // namespace snap

namespace popart {
class Op;

namespace popx {
class Devicex;

class TensorRemapOpx : public PopOpx {
public:
  TensorRemapOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
  bool outputCreatedExternally(OutIndex) const final;
  InputCreatorType getInputCreatorType(InIndex index) const final;
  snap::Tensor unwindTensorLayout(snap::Tensor, InIndex, OutIndex) const final;
  view::RegMap unwindRegion(InIndex, OutIndex) const final;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_TENSORREMAPX_HPP_
