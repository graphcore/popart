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

  auto lossScale        = getInTensor(op.getLossScaleInIndex());
  auto inverseLossScale = getInTensor(op.getInverseLossScaleInIndex());

  // Check there is at least one 'gradient statistics' input
  if (!hasInput(1)) {
    throw error("LossScaleUpdateOpx {} does not have any input at InIndex 1",
                op.str());
  }

  float thresholdUpperCountProportion = 0.2;
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

  poplar::program::Sequence scaleUp;
  popops::mulInPlace(graph(), lossScale, 2.0, scaleUp, debugContext("scaleUp"));
  popops::mulInPlace(
      graph(), inverseLossScale, 0.5, scaleUp, debugContext("scaleUp"));

  poplar::program::Sequence scaleDown;
  popops::mulInPlace(
      graph(), lossScale, 0.5, scaleDown, debugContext("scaleDown"));
  popops::mulInPlace(
      graph(), inverseLossScale, 2.0, scaleDown, debugContext("scaleDown"));

  prog.add(poplar::program::If(
      shouldScaleDown, scaleDown, scaleUp, debugContext("lossScaleUpdate")));

  setOutTensor(op.getUpdatedLossScaleOutIndex(), lossScale);
  setOutTensor(op.getUpdatedInverseLossScaleOutIndex(), inverseLossScale);
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
