// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_LOOPX_HPP
#define GUARD_NEURALNET_LOOPX_HPP

#include <popart/popx/opx.hpp>

namespace popart {
namespace popx {

class LoopOpx : public Opx {
public:
  LoopOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

private:
  void copyOpInputsToBodyInputs(poplar::program::Sequence &prog) const;
  void copyBodyOutputsToExplicitBodyInputs(
      poplar::program::Sequence &prog,
      const std::vector<poplar::Tensor> &bodyOutputs) const;
  void copyBodyOutputsToOpOutputs(
      poplar::program::Sequence &prog,
      const std::vector<poplar::Tensor> &bodyOutputs) const;
  std::vector<poplar::Tensor> prepareBodyOutputs() const;
};
} // namespace popx
} // namespace popart

#endif
