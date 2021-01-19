// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_REVERSEX_HPP
#define GUARD_NEURALNET_REVERSEX_HPP

#include <popart/popx/opx.hpp>

namespace popart {
namespace popx {

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

#endif
