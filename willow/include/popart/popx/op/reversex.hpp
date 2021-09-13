// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_REVERSEX_HPP
#define GUARD_NEURALNET_REVERSEX_HPP

#include <popart/popx/popopx.hpp>

namespace popart {
namespace popx {

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

#endif
