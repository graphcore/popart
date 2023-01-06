// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_TENSORREMAPX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_TENSORREMAPX_HPP_

#include <poplar/Tensor.hpp>
#include <popart/names.hpp>
#include <popart/popx/opx.hpp>

namespace poplar {
namespace program {
class Sequence;
} // namespace program
} // namespace poplar

namespace popart {
class Op;

namespace popx {
class Devicex;

class TensorRemapOpx : public Opx {
public:
  TensorRemapOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
  bool outputCreatedExternally(OutIndex) const final;
  InputCreatorType getInputCreatorType(InIndex index) const final;
  poplar::Tensor
      unwindTensorLayout(poplar::Tensor, InIndex, OutIndex) const final;
  view::RegMap unwindRegion(InIndex, OutIndex) const final;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_TENSORREMAPX_HPP_
