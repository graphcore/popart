// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_SCALEDVARUPDATEX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_SCALEDVARUPDATEX_HPP_

#include <poplar/Tensor.hpp>
#include <popart/popx/op/varupdatex.hpp>

namespace poplar {
namespace program {
class Sequence;
} // namespace program
} // namespace poplar

namespace popart {
class ScaledVarUpdateOp;
class Op;

namespace popx {
class Devicex;

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

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_SCALEDVARUPDATEX_HPP_
