// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ADAPTIVEVARUPDATEX_HPP
#define GUARD_NEURALNET_ADAPTIVEVARUPDATEX_HPP

#include <popart/popx/op/varupdatex.hpp>

namespace popart {
namespace popx {

class ScaledVarUpdateOpx : public VarUpdateOpx {
public:
  ScaledVarUpdateOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

  // does not create inputs
private:
  poplar::Tensor getOrCreateLrTensor() const;
  poplar::Tensor getOrCreateWdTensor() const;

  void growWithLrAsInput(poplar::program::Sequence &prog,
                         const ScaledVarUpdateOp &op,
                         const poplar::Tensor &var,
                         const poplar::Tensor &updater) const;
  void growWithLrInUpdater(poplar::program::Sequence &prog,
                           const ScaledVarUpdateOp &op,
                           const poplar::Tensor &var,
                           const poplar::Tensor &updater) const;
};

} // namespace popx
} // namespace popart

#endif
