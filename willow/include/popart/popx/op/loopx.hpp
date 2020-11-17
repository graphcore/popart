// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_LOOPX_HPP
#define GUARD_NEURALNET_LOOPX_HPP

#include <popart/popx/op/subgraphx.hpp>
#include <popart/popx/opx.hpp>

namespace popart {
namespace popx {

class LoopOpx : public SubgraphOpx {
public:
  LoopOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;
  InputCreatorType getInputCreatorType(InIndex) const;
  bool canUnwind(InIndex in, OutIndex out) const final;
  poplar::Tensor
      unwindTensorLayout(poplar::Tensor, InIndex, OutIndex) const final;
  view::RegMap unwindRegion(InIndex, OutIndex) const final;

private:
  void copyExplicitOpInputsToBodyOutputs(poplar::program::Sequence &prog) const;
  void copyImplicitOpInputsToImplicitBodyInputs(
      poplar::program::Sequence &prog) const;
  void
  copyBodyOutputsToExplicitBodyInputs(poplar::program::Sequence &prog) const;
  void copyBodyOutputsToOpOutputs(poplar::program::Sequence &prog) const;
};
} // namespace popx
} // namespace popart

#endif
