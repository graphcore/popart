// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_DYNAMICSLICEX_HPP
#define GUARD_NEURALNET_DYNAMICSLICEX_HPP

#include <popart/names.hpp>
#include <popart/popx/popopx.hpp>

namespace popart {
namespace popx {

class DynamicSliceOpx : public PopOpx {
public:
  DynamicSliceOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;
  InputCreatorType getInputCreatorType(InIndex index) const final;
  snap::Tensor unwindTensorLayout(snap::Tensor, InIndex, OutIndex) const final;
  view::RegMap unwindRegion(InIndex, OutIndex) const final;
  snap::Tensor
  createInputTensor(InIndex index,
                    const poplar::DebugNameAndId &dnai) const final;
  std::set<TensorId> mustExistBeforeCreate(InIndex) const final { return {}; }
};

} // namespace popx
} // namespace popart

#endif
