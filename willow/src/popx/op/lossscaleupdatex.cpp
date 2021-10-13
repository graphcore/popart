// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <limits>

#include <poplar/Tensor.hpp>

#include <popart/op/lossscaleupdate.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/lossscaleupdatex.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/util.hpp>

#include <popops/Cast.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Zero.hpp>

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

  if (op.input->n() != 3) {
    throw error("LossScaleUpdateOpx has {} inputs, but 3 inputs are expected.",
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

  auto gradStats =
      getInTensor(op.getStatisticsTensorInIndex()).getPoplarTensor();
  auto lowerBinCount = gradStats.slice(0, 1, 0).reshape({});
  auto upperBinCount = gradStats.slice(1, 2, 0).reshape({});

  lowerBinCount = popops::cast(graph().getPoplarGraph(),
                               lowerBinCount,
                               poplar::FLOAT,
                               prog.getPoplarSequence(),
                               debugContext());
  upperBinCount = popops::cast(graph().getPoplarGraph(),
                               upperBinCount,
                               poplar::FLOAT,
                               prog.getPoplarSequence(),
                               debugContext());
  popops::mulInPlace(graph().getPoplarGraph(),
                     lowerBinCount,
                     f,
                     prog.getPoplarSequence(),
                     debugContext("scaleSumLowerBinCounts"));

  auto shouldScaleDown = popops::map(graph().getPoplarGraph(),
                                     popops::expr::BinaryOpType::GREATER_THAN,
                                     upperBinCount,
                                     lowerBinCount,
                                     prog.getPoplarSequence(),
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

  prog.getPoplarSequence().add(
      poplar::program::If(shouldScaleDown,
                          scaleDown.getPoplarSequence(),
                          scaleUp.getPoplarSequence(),
                          debugContext("lossScaleUpdate")));
  if (op.getClipOutput()) {
    // Whenever the finalLossScale is in fp16 or the weights are in fp16, the
    // finalLossScale should be clipped so that its value fits in the fp16
    // range. We chose to clip it at the largest power of 2 that fits in fp16 -
    // (2^15 = 32768). The finalLossScale is calculated as:
    //     finalLossScale = lossScaleUpdateFactor * lossScaling
    // As a result, the lossScaleUpdateFactor should be clipped at
    // 2^15 / lossScaling to satisfy the requirement above.
    auto lossScaling =
        getInTensor(op.getLossScalingInIndex()).getPoplarTensor();

    auto clipAt_ = graph().getPoplarGraph().addConstant<float>(
        lossScaling.elementType(), {1}, std::numeric_limits<short>::max() + 1);
    graph().getPoplarGraph().setTileMapping(clipAt_, 0);

    auto clipAt = popops::div(graph().getPoplarGraph(),
                              clipAt_,
                              lossScaling,
                              prog.getPoplarSequence(),
                              debugContext("clipAtValue"));
    popops::minInPlace(graph().getPoplarGraph(),
                       lossScaleUpdateFactor,
                       clipAt,
                       prog.getPoplarSequence(),
                       debugContext("clipOutput"));
  }

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
