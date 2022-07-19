// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_CTCX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_CTCX_HPP_

#include "popart/popx/debugcontextx.hpp"
#include <memory>
#include <set>
#include <snap/Tensor.hpp>
#include <popart/names.hpp>
#include <popart/popx/popopx.hpp>

namespace snap {
namespace program {
class Sequence;
} // namespace program
} // namespace snap

namespace popnn {
namespace ctc {
class Plan;
}
} // namespace popnn

namespace popart {
class Op;

namespace popx {
class Devicex;

class CtcOpx : public PopOpx {
public:
  CtcOpx(Op *, Devicex *);
  ~CtcOpx();
  void grow(snap::program::Sequence &) const final;

  // See PopOpx::createInputTensor.
  virtual snap::Tensor
  createInputTensor(InIndex index,
                    const poplar::DebugNameAndId &dnai) const override;

  // See PopOpx::getInputCreatorType.
  virtual InputCreatorType getInputCreatorType(InIndex index) const override;

  // See PopOpx::mustExistBeforeCreate.
  std::set<TensorId> mustExistBeforeCreate(InIndex index) const override;

private:
  // Helper function to apply reduction.
  snap::Tensor applyReduction(snap::program::Sequence &prog,
                              snap::Tensor loss,
                              snap::Tensor targetLengths) const;

  // Unique pointer (so we can forward-declare and avoid including poplar
  // headers).
  std::unique_ptr<popnn::ctc::Plan> plan;
};

class CtcGradOpx : public PopOpx {
public:
  CtcGradOpx(Op *, Devicex *);
  void grow(snap::program::Sequence &) const final;

private:
  // Helper function to apply partial derivative of reduction.
  snap::Tensor applyReductionGrad(snap::program::Sequence &prog,
                                  const snap::Tensor &ctcLossGrad,
                                  const snap::Tensor &targetLengths) const;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_CTCX_HPP_
