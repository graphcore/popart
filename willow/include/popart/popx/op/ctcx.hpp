// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_CTCX_HPP
#define GUARD_NEURALNET_CTCX_HPP

#include <memory>

#include <popart/names.hpp>
#include <popart/op/identity.hpp>
#include <popart/popx/popopx.hpp>

namespace popnn {
namespace ctc {
class Plan;
}
} // namespace popnn

namespace popart {
namespace popx {

class CtcOpx : public PopOpx {
public:
  CtcOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

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
  snap::Tensor applyReduction(poplar::program::Sequence &prog,
                              snap::Tensor loss,
                              snap::Tensor targetLengths) const;

  // Unique pointer (so we can forward-declare and avoid including poplar
  // headers).
  std::unique_ptr<popnn::ctc::Plan> plan;
};

class CtcGradOpx : public PopOpx {
public:
  CtcGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

private:
  // Helper function to apply partial derivative of reduction.
  snap::Tensor applyReductionGrad(poplar::program::Sequence &prog,
                                  const snap::Tensor &ctcLossGrad,
                                  const snap::Tensor &targetLengths) const;
};

} // namespace popx
} // namespace popart

#endif
