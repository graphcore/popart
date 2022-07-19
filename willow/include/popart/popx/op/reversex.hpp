// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_REVERSEX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_REVERSEX_HPP_

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

class ReverseBaseOpx : public PopOpx {
public:
  ReverseBaseOpx(Op *, Devicex *);
  InputCreatorType getInputCreatorType(InIndex) const final {
    return InputCreatorType::CanUnwind;
  }
  snap::Tensor unwindTensorLayout(snap::Tensor tensor,
                                  InIndex inIndex,
                                  OutIndex outIndex) const final;
  view::RegMap unwindRegion(InIndex inIndex, OutIndex outIndex) const final;
};

class ReverseOpx : public ReverseBaseOpx {
public:
  ReverseOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
};

class ReverseInplaceOpx : public ReverseBaseOpx {
public:
  ReverseInplaceOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
};

// The gradient of a reverse is also a reverse
class ReverseGradOpx : public ReverseOpx {
public:
  ReverseGradOpx(Op *, Devicex *);
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_REVERSEX_HPP_
