// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ADADELTAUPDATERX_HPP
#define GUARD_NEURALNET_ADADELTAUPDATERX_HPP

#include <popart/popx/opx.hpp>

namespace popart {
namespace popx {

class AdaDeltaUpdaterOpx : public Opx {
public:
  AdaDeltaUpdaterOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

  // can create the accumulator2 input Tensor (@Accl2 index)
  // from the weight gradient tensor (@Accl1 index)
  poplar::Tensor createInput(InIndex,
                             const poplar::DebugNameAndId &dnai) const final;
  InputCreatorType getInputCreatorType(InIndex) const final;
  std::vector<TensorId> mustExistBeforeCreate(InIndex) const final;
  bool hasCreatorViewChangers(InIndex index) const final;
  ViewChangers getCreatorViewChangers(InIndex index) const final;
};

} // namespace popx
} // namespace popart

#endif
