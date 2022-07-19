// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_SCALEDVARUPDATEX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_SCALEDVARUPDATEX_HPP_

#include <snap/Tensor.hpp>
#include <popart/popx/op/varupdatex.hpp>

namespace snap {
namespace program {
class Sequence;
} // namespace program
} // namespace snap

namespace popart {
class ScaledVarUpdateOp;
class Op;

namespace popx {
class Devicex;

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

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_SCALEDVARUPDATEX_HPP_
