// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_CTCX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_CTCX_HPP_

#include <memory>
#include <set>
#include <poplar/Tensor.hpp>
#include <popart/names.hpp>
#include <popart/popx/opx.hpp>

#include "popart/popx/debugcontextx.hpp"

namespace poplar {
namespace program {
class Sequence;
} // namespace program
} // namespace poplar

namespace popnn {
namespace ctc {
class Plan;
}
} // namespace popnn

namespace popart {
class Op;

namespace popx {
class Devicex;

class CtcOpx : public Opx {
public:
  CtcOpx(Op *, Devicex *);
  ~CtcOpx();
  void grow(poplar::program::Sequence &) const final;

  // See Opx::createInput.
  poplar::Tensor createInput(InIndex index,
                             const poplar::DebugNameAndId &dnai) const override;

  // See Opx::getInputCreatorType.
  InputCreatorType getInputCreatorType(InIndex index) const override;

  // See Opx::mustExistBeforeCreate.
  std::set<TensorId> mustExistBeforeCreate(InIndex index) const override;

private:
  // Helper function to apply reduction.
  poplar::Tensor applyReduction(poplar::program::Sequence &prog,
                                poplar::Tensor loss,
                                poplar::Tensor targetLengths) const;

  // Unique pointer (so we can forward-declare and avoid including poplar
  // headers).
  std::unique_ptr<popnn::ctc::Plan> plan;
};

class CtcGradOpx : public Opx {
public:
  CtcGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

private:
  // Helper function to apply partial derivative of reduction.
  poplar::Tensor applyReductionGrad(poplar::program::Sequence &prog,
                                    const poplar::Tensor &ctcLossGrad,
                                    const poplar::Tensor &targetLengths) const;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_CTCX_HPP_
