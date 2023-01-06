// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <memory>
#include <poplar/Program.hpp>
#include <poplar/Tensor.hpp>
#include <poplar/Type.hpp>
#include <popops/Cast.hpp>
#include <popops/ElementWise.hpp>
#include <popops/ExprOp.hpp>
#include <popart/op/lossscaleupdate.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/lossscaleupdatex.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/error.hpp"
#include "popart/graphcoreoperators.hpp"
#include "popart/ir.hpp"
#include "popart/logging.hpp"
#include "popart/op.hpp"
#include "popart/popx/opx.hpp"
#include "popart/sessionoptions.hpp"
#include "popart/tensorindex.hpp"

namespace popart {
namespace popx {

void LossScaleUpdateOpx::grow(poplar::program::Sequence &prog) const {
  auto &op = getOp<LossScaleUpdateOp>();
  auto &ir = op_p->getIr();

  // Check there is at least one 'gradient statistics' input
  if (!hasInput(op.getStatisticsTensorInIndex())) {
    throw error("LossScaleUpdateOpx {} does not have any input at InIndex {}",
                op.str(),
                op.getStatisticsTensorInIndex());
  }

  if (op.input->n() != 2) {
    throw error("LossScaleUpdateOpx has {} inputs, but 2 inputs are expected.",
                op.input->n());
  }

  // Get automatic loss scaling hyperparameters.
  float thresholdUpperCountProportion =
      ir.getSessionOptions()
          .automaticLossScalingSettings.thresholdUpperCountProportion;
  if (thresholdUpperCountProportion < 0 || thresholdUpperCountProportion > 1) {
    throw error("Out of range value for 'thresholdUpperCountProportion'. The "
                "current value is {}, but it should be in the range [0, 1].",
                thresholdUpperCountProportion);
  }
  // If the upper bin counts (u), as a proportion of the total
  // (lower (l) + upper) bin counts, is greater than the threshold
  // (t), divide the loss scaling factor by 2. Else, multiply the
  // loss scaling factor by 2.
  // i.e.:
  // new_ls = old_ls / 2  if u/(u + l) > t
  //          old_ls * 2  otherwise
  //
  // This can also be expressed as:
  // new_ls = old_ls / 2  if u > f * l
  //          old_ls * 2  otherwise
  // where f = t/(1-t)
  float f = thresholdUpperCountProportion / (1 - thresholdUpperCountProportion);

  auto gradStats     = getInTensor(op.getStatisticsTensorInIndex());
  auto lowerBinCount = gradStats.slice(0, 1, 0).reshape({});
  auto upperBinCount = gradStats.slice(1, 2, 0).reshape({});

  lowerBinCount =
      popops::cast(graph(), lowerBinCount, poplar::FLOAT, prog, debugContext());
  upperBinCount =
      popops::cast(graph(), upperBinCount, poplar::FLOAT, prog, debugContext());
  popops::mulInPlace(
      graph(), lowerBinCount, f, prog, debugContext("scaleSumLowerBinCounts"));

  auto shouldScaleDown = popops::map(graph(),
                                     popops::expr::BinaryOpType::GREATER_THAN,
                                     upperBinCount,
                                     lowerBinCount,
                                     prog,
                                     debugContext());

  auto lossScaleUpdateFactor =
      getInTensor(op.getLossScaleUpdateFactorInIndex());
  auto updateFactorDType = popType(op.getUpdateFactorDType());
  poplar::program::Sequence scaleUp, scaleDown;
  popops::mulInPlace(
      graph(), lossScaleUpdateFactor, 2.0, scaleUp, debugContext("scaleUp"));
  popops::mulInPlace(graph(),
                     lossScaleUpdateFactor,
                     0.5,
                     scaleDown,
                     debugContext("scaleDown"));

  prog.add(poplar::program::If(
      shouldScaleDown, scaleDown, scaleUp, debugContext("lossScaleUpdate")));

  setOutTensor(op.getUpdatedLossScaleUpdateFactorOutIndex(),
               lossScaleUpdateFactor);
}

LossScaleUpdateOpx::LossScaleUpdateOpx(Op *op, Devicex *devicex)
    : Opx(op, devicex) {
  verifyOp<LossScaleUpdateOp>(op);
}

namespace {
OpxCreator<LossScaleUpdateOpx>
    lossScaleUpdateOpxCreator(Onnx::CustomOperators::LossScaleUpdate);
} // namespace

} // namespace popx
} // namespace popart
