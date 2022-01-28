// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_TENSORREMAPX_HPP
#define GUARD_NEURALNET_TENSORREMAPX_HPP

#include <popart/names.hpp>

namespace popart {
namespace popx {

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

#endif
