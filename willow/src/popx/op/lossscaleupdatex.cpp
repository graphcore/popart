// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <memory>
#include <snap/Graph.hpp>
#include <snap/Program.hpp>
#include <snap/Tensor.hpp>
#include <snap/popops/ElementWise.hpp>
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
#include "popart/popx/popopx.hpp"
#include "popart/sessionoptions.hpp"
#include "popart/tensorindex.hpp"

namespace popart {
namespace popx {

void LossScaleUpdateOpx::grow(snap::program::Sequence &prog) const {
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

  lowerBinCount = snap::Tensor{popops::cast(graph().getPoplarGraph(),
                                            lowerBinCount.getPoplarTensor(),
                                            poplar::FLOAT,
                                            prog.getPoplarSequence(),
                                            debugContext()),
                               graph()};
  upperBinCount = snap::Tensor{popops::cast(graph().getPoplarGraph(),
                                            upperBinCount.getPoplarTensor(),
                                            poplar::FLOAT,
                                            prog.getPoplarSequence(),
                                            debugContext()),
                               graph()};
  popops::mulInPlace(graph().getPoplarGraph(),
                     lowerBinCount.getPoplarTensor(),
                     f,
                     prog.getPoplarSequence(),
                     debugContext("scaleSumLowerBinCounts"));

  auto shouldScaleDown =
      snap::popops::map(graph(),
                        popops::expr::BinaryOpType::GREATER_THAN,
                        upperBinCount,
                        lowerBinCount,
                        prog,
                        debugContext());

  auto lossScaleUpdateFactor =
      getInTensor(op.getLossScaleUpdateFactorInIndex()).getPoplarTensor();
  auto updateFactorDType = popType(op.getUpdateFactorDType());
  snap::program::Sequence scaleUp(graph()), scaleDown(graph());
  popops::mulInPlace(graph().getPoplarGraph(),
                     lossScaleUpdateFactor,
                     2.0,
                     scaleUp.getPoplarSequence(),
                     debugContext("scaleUp"));
  popops::mulInPlace(graph().getPoplarGraph(),
                     lossScaleUpdateFactor,
                     0.5,
                     scaleDown.getPoplarSequence(),
                     debugContext("scaleDown"));

  prog.add(snap::program::If(
      shouldScaleDown, scaleDown, scaleUp, debugContext("lossScaleUpdate")));

  setOutTensor(op.getUpdatedLossScaleUpdateFactorOutIndex(),
               snap::Tensor{lossScaleUpdateFactor, graph()});
}

LossScaleUpdateOpx::LossScaleUpdateOpx(Op *op, Devicex *devicex)
    : PopOpx(op, devicex) {
  verifyOp<LossScaleUpdateOp>(op);
}

namespace {
OpxCreator<LossScaleUpdateOpx>
    lossScaleUpdateOpxCreator(Onnx::CustomOperators::LossScaleUpdate);
} // namespace

} // namespace popx
} // namespace popart
