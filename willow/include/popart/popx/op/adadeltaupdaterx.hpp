// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ADADELTAUPDATERX_HPP
#define GUARD_NEURALNET_ADADELTAUPDATERX_HPP

#include <popart/popx/popopx.hpp>

namespace popart {
namespace popx {

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

#endif
