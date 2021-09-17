// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_SUMX_HPP
#define GUARD_NEURALNET_SUMX_HPP

#include <popart/names.hpp>
#include <popart/popx/popopx.hpp>

namespace popart {
namespace popx {

class SumOpx : public PopOpx {
public:
  SumOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
  InputCreatorType getInputCreatorType(InIndex) const final;
  snap::Tensor unwindTensorLayout(snap::Tensor tensor,
                                  InIndex inIndex,
                                  OutIndex outIndex) const final;
  view::RegMap unwindRegion(InIndex, OutIndex) const final;
};

class SumArgGradOpx : public PopOpx {
public:
  SumArgGradOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
};

} // namespace popx
} // namespace popart

#endif
