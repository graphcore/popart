// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_REVERSEX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_REVERSEX_HPP_

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

class ReverseBaseOpx : public Opx {
public:
  ReverseBaseOpx(Op *, Devicex *);
  InputCreatorType getInputCreatorType(InIndex) const final {
    return InputCreatorType::CanUnwind;
  }
  poplar::Tensor unwindTensorLayout(poplar::Tensor tensor,
                                    InIndex inIndex,
                                    OutIndex outIndex) const final;
  view::RegMap unwindRegion(InIndex inIndex, OutIndex outIndex) const final;
};

class ReverseOpx : public ReverseBaseOpx {
public:
  ReverseOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

class ReverseInplaceOpx : public ReverseBaseOpx {
public:
  ReverseInplaceOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
};

// The gradient of a reverse is also a reverse
class ReverseGradOpx : public ReverseOpx {
public:
  ReverseGradOpx(Op *, Devicex *);
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_REVERSEX_HPP_
