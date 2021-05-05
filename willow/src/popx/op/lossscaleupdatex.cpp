// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <popart/op/lossscaleupdate.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/lossscaleupdatex.hpp>
#include <popart/popx/opxmanager.hpp>

#include <popops/Cast.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Zero.hpp>

namespace popart {
namespace popx {

void LossScaleUpdateOpx::grow(poplar::program::Sequence &prog) const {
  auto &op = getOp<LossScaleUpdateOp>();
  auto &ir = op_p->getIr();

  // Check there is at least one 'gradient statistics' input
  if (!hasInput(op.getFirstStatisticsTensorInIndex())) {
    throw error("LossScaleUpdateOpx {} does not have any input at InIndex 1",
                op.str());
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

  auto sumLowerBinCounts = getScalarVariable(
      popType(op.inInfo(op.getFirstStatisticsTensorInIndex()).dataType()),
      "sumLowerBinCounts");
  auto sumUpperBinCounts = getScalarVariable(
      popType(op.inInfo(op.getFirstStatisticsTensorInIndex()).dataType()),
      "sumUpperBinCounts");
  popops::zero(
      graph(), sumLowerBinCounts, prog, debugContext("zeroLowerBinCounts"));
  popops::zero(
      graph(), sumUpperBinCounts, prog, debugContext("zeroUpperBinCounts"));

  for (int i = op.getFirstStatisticsTensorInIndex(); i < op.input->n(); i++) {
    auto gradStats     = getInTensor(i);
    auto lowerBinCount = gradStats.slice(0, 1, 0);
    auto upperBinCount = gradStats.slice(1, 2, 0);

    popops::addInPlace(graph(),
                       sumLowerBinCounts,
                       lowerBinCount,
                       prog,
                       debugContext("sumLowerBinCounts"));
    popops::addInPlace(graph(),
                       sumUpperBinCounts,
                       upperBinCount,
                       prog,
                       debugContext("sumUpperBinCounts"));
  }

  sumLowerBinCounts = popops::cast(
      graph(), sumLowerBinCounts, poplar::FLOAT, prog, debugContext());
  sumUpperBinCounts = popops::cast(
      graph(), sumUpperBinCounts, poplar::FLOAT, prog, debugContext());
  popops::mulInPlace(graph(),
                     sumLowerBinCounts,
                     f,
                     prog,
                     debugContext("scaleSumLowerBinCounts"));

  auto shouldScaleDown = popops::map(graph(),
                                     popops::expr::BinaryOpType::GREATER_THAN,
                                     sumUpperBinCounts,
                                     sumLowerBinCounts,
                                     prog,
                                     debugContext());

  auto updateFactorDType = popType(op.getUpdateFactorDType());
  auto scaleUpTensor     = getConst(updateFactorDType, {}, 2.0, "scaleUp");
  auto scaleDownTensor   = getConst(updateFactorDType, {}, 0.5, "scaledown");
  auto updateFactor      = getScalarVariable(updateFactorDType, "updateFactor");
  poplar::program::Copy scaleUp(
      scaleUpTensor, updateFactor, false, debugContext("scaleDown"));
  poplar::program::Copy scaleDown(
      scaleDownTensor, updateFactor, false, debugContext("scaleDown"));

  prog.add(poplar::program::If(
      shouldScaleDown, scaleDown, scaleUp, debugContext("lossScaleUpdate")));

  setOutTensor(op.getLossScaleUpdateFactorOutIndex(), updateFactor);
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
