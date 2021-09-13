// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ADAPTIVEVARUPDATEX_HPP
#define GUARD_NEURALNET_ADAPTIVEVARUPDATEX_HPP

#include <popart/popx/op/varupdatex.hpp>

namespace popart {
namespace popx {

class ScaledVarUpdateOpx : public VarUpdateOpx {
public:
  ScaledVarUpdateOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;

  // does not create inputs
private:
  snap::Tensor getOrCreateLrTensor() const;
  snap::Tensor getOrCreateWdTensor() const;

  void growWithLrAsInput(snap::program::Sequence &prog,
                         const ScaledVarUpdateOp &op,
                         const snap::Tensor &var,
                         const snap::Tensor &updater) const;
  void growWithLrInUpdater(snap::program::Sequence &prog,
                           const ScaledVarUpdateOp &op,
                           const snap::Tensor &var,
                           const snap::Tensor &updater) const;
};

} // namespace popx
} // namespace popart

#endif
