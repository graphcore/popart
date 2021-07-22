// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <memory>
#include <popart/error.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/op/ctc.hpp>
#include <popart/optimizer.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/ctcx.hpp>
#include <popart/popx/opxmanager.hpp>

#include <popnn/CTCLoss.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Expr.hpp>
#include <popops/Reduce.hpp>

namespace pe = popops::expr;

namespace popart {
namespace popx {

CtcOpx::CtcOpx(Op *op_, Devicex *devicex)
    : PopOpx(op_, devicex), plan(std::make_unique<popnn::ctc::Plan>()) {
  verifyOp<CtcOp>(op_, Onnx::CustomOperators::Ctc);

  const auto &op            = getOp<CtcOp>();
  auto logProbsPopartTensor = op.input->tensor(CtcOp::getLogProbsInIndex());
  auto ctcLossPopartTensor  = op.output->tensor(CtcOp::getCtcLossOutIndex());
  auto inDtype  = popType(logProbsPopartTensor->info.getDataTypeInfo()->type());
  auto outDtype = popType(ctcLossPopartTensor->info.getDataTypeInfo()->type());

  // Create plan once and re-use for growing and createInput.
  *plan = popnn::ctc::plan(graph().getPoplarGraph(),
                           inDtype,
                           outDtype,
                           op.getBatchSize(),
                           op.getMaxInputLength(),
                           op.getMaxTargetLength(),
                           op.getNumClasses(),
                           {});
}

void CtcOpx::grow(poplar::program::Sequence &prog) const {

  const auto &op = getOp<CtcOp>();
  const auto &ir = op.getIr();

  if (!ir.isTraining()) {
    throw error("CtcOpx is currently not supported for infererence sessions");
  } else {

    const auto &op           = getOp<CtcOp>();
    auto ctcLossPopartTensor = op.output->tensor(CtcOp::getCtcLossOutIndex());
    auto outDtype =
        popType(ctcLossPopartTensor->info.getDataTypeInfo()->type());

    const auto &logProbs      = getInTensor(CtcOp::getLogProbsInIndex());
    const auto &targets       = getInTensor(CtcOp::getTargetsInIndex());
    const auto &inputLengths  = getInTensor(CtcOp::getInputLengthsInIndex());
    const auto &targetLengths = getInTensor(CtcOp::getTargetLengthsInIndex());

    auto result = popnn::ctc::calcLossAndGradientLogProbabilities(
        graph().getPoplarGraph(),
        outDtype,
        logProbs.getPoplarTensor(),
        targets.getPoplarTensor(),
        inputLengths.getPoplarTensor(),
        targetLengths.getPoplarTensor(),
        prog,
        op.getBlank(),
        *plan,
        debugContext("lossAndGrad"));

    snap::Tensor ctcLoss = snap::Tensor{result.first, graph()};

    ctcLoss = applyReduction(prog, ctcLoss, targetLengths);

    setOutTensor(CtcOp::getCtcLossOutIndex(), ctcLoss);
    setOutTensor(CtcOp::getLogProbsGradientWrtCtcLossOutIndex(),
                 snap::Tensor{result.second, graph()});
  }
}

snap::Tensor
CtcOpx::createInputTensor(InIndex index,
                          const poplar::DebugNameAndId &dnai) const {
  const auto &op       = getOp<CtcOp>();
  auto logProbsTensor  = op.input->tensor(CtcOp::getLogProbsInIndex());
  auto targetsInTensor = op.input->tensor(CtcOp::getTargetsInIndex());
  auto logProbsDtype = popType(logProbsTensor->info.getDataTypeInfo()->type());
  auto targetsDtype  = popType(targetsInTensor->info.getDataTypeInfo()->type());
  auto maxInputLen   = logProbsTensor->info.dim(0);
  auto batchSize     = logProbsTensor->info.dim(1);
  auto numClasses    = logProbsTensor->info.dim(2);
  auto maxTargetLen  = targetsInTensor->info.dim(1);

  if (index == CtcOp::getLogProbsInIndex()) {

    return snap::Tensor{popnn::ctc::createDataInput(graph().getPoplarGraph(),
                                                    logProbsDtype,
                                                    batchSize,
                                                    maxInputLen,
                                                    numClasses,
                                                    *plan,
                                                    dnai),
                        graph()};

  } else if (index == CtcOp::getTargetsInIndex()) {

    return snap::Tensor{popnn::ctc::createLabelsInput(graph().getPoplarGraph(),
                                                      targetsDtype,
                                                      batchSize,
                                                      maxTargetLen,
                                                      *plan,
                                                      dnai),
                        graph()};

  } else {
    throw error("CtcOpx::createInput : Invalid index = " +
                std::to_string(index));
  }
}

InputCreatorType CtcOpx::getInputCreatorType(InIndex index) const {
  if (index == CtcOp::getLogProbsInIndex()) {
    return InputCreatorType::CanCreate;
  } else if (index == CtcOp::getTargetsInIndex()) {
    return InputCreatorType::CanCreate;
  } else {
    return InputCreatorType::Deadend;
  }
}

std::set<TensorId> CtcOpx::mustExistBeforeCreate(InIndex index) const {

  if (index == CtcOp::getLogProbsInIndex() ||
      index == CtcOp::getTargetsInIndex()) {
    return {};
  } else {
    throw error("CtcOpx::mustExistBeforeCreate : Invalid index = " +
                std::to_string(index));
  }
}

snap::Tensor CtcOpx::applyReduction(poplar::program::Sequence &prog,
                                    snap::Tensor ctcLoss,
                                    snap::Tensor targetLengths) const {

  const auto &op = getOp<CtcOp>();

  if (op.getReductionType() == ReductionType::NoReduction) {
    // No reduction required.
    return ctcLoss;

  } else {
    // Reduction is required.
    auto inTensor1D = ctcLoss.getPoplarTensor().flatten();

    double scale = 0.;
    switch (op.getReductionType()) {
    case ReductionType::Sum: {
      // No scale.
      scale = 1.0;
      break;
    }
    case ReductionType::Mean: {
      // Divide by target max(length, 1).
      ctcLoss = snap::Tensor{
          popops::map(
              graph().getPoplarGraph(),
              pe::Divide(pe::_1,
                         pe::Cast(pe::Max(pe::_2, pe::Const(1)),
                                  ctcLoss.elementType())),
              {ctcLoss.getPoplarTensor(), targetLengths.getPoplarTensor()},
              prog,
              debugContext("divByTargetLen")),
          graph()};

      // Take the average.
      double totalSamples = static_cast<double>(ctcLoss.dim(0));
      scale               = 1.0 / totalSamples;
      break;
    }
    case ReductionType::NoReduction:
    default: {
      throw error("Unsupported reduction type for Loss {}",
                  debugContext().getPathName());
    }
    }

    // Always expected to be FLOAT, regardless of the input type.
    auto t_scale = getConst(poplar::FLOAT, {}, scale, "scale");

    // Do the reduction.
    ctcLoss = snap::Tensor{
        popops::reduce(
            graph().getPoplarGraph(),
            ctcLoss.getPoplarTensor(),
            {0},
            {popops::Operation::ADD, false, t_scale.getPoplarTensor()},
            prog,
            debugContext("reduce")),
        graph()};
  }

  return ctcLoss;
}

CtcGradOpx::CtcGradOpx(Op *op, Devicex *devicex) : PopOpx(op, devicex) {
  verifyOp<CtcGradOp>(op, Onnx::CustomGradOperators::CtcGrad);
}

void CtcGradOpx::grow(poplar::program::Sequence &prog) const {

  const CtcGradOp &gradOp = getOp<CtcGradOp>();

  // Get dimensions for T and C.
  auto outTensor =
      gradOp.output->tensor(CtcGradOp::getLogProbsGradientOutIndex());
  const auto &outShape = outTensor->info.shape();
  const unsigned T     = outShape[0];
  const unsigned C     = outShape[2];

  // Should be shape [N].
  const snap::Tensor &targetLengths =
      getInTensor(CtcGradOp::getTargetLengthsInIndex());

  // Should be shape [T, N, C].
  const snap::Tensor &logProbsGradientWrtCtcLoss =
      getInTensor(CtcGradOp::getLogProbsGradientWrtCtcLossInIndex());

  // Shape [] if reduction else shape [N].
  const snap::Tensor &ctcLossGrad =
      getInTensor(CtcGradOp::getCtcLossGradientInIndex());

  // Apply chain rule for reduction. The result gradient tensor is shape [N].
  auto adjustedCtcLossGrad =
      applyReductionGrad(prog, ctcLossGrad, targetLengths).getPoplarTensor();

  // Expand and broadcast the gradient tensor to [T,N,C].
  adjustedCtcLossGrad =
      adjustedCtcLossGrad.expand({0, 1}).broadcast(T, 0).broadcast(C, 2);

  // Apply chain rule for CTC loss.
  auto inType  = adjustedCtcLossGrad.elementType();
  auto outType = popType(outTensor->info.getDataTypeInfo()->type());

  // If the output type differs from the input, do something.
  auto expr = (inType == outType) ? pe::Mul(pe::_1, pe::_2)
                                  : pe::Mul(pe::Cast(pe::_1, outType),
                                            pe::Cast(pe::_2, outType));

  auto logProbsGradient = popops::map(
      graph().getPoplarGraph(),
      expr,
      {logProbsGradientWrtCtcLoss.getPoplarTensor(), adjustedCtcLossGrad},
      prog,
      debugContext("chainRule"));

  setOutTensor(CtcGradOp::getLogProbsGradientOutIndex(),
               snap::Tensor{logProbsGradient, graph()});
}

snap::Tensor
CtcGradOpx::applyReductionGrad(poplar::program::Sequence &prog,
                               const snap::Tensor &ctcLossGrad,
                               const snap::Tensor &targetLengths) const {

  // In the forward pass we the loss output of the CTC loss function outputs
  // loss tensor of size [N]. Depending on reduction settings, we applied either
  // applied no reduction, a sum reduction or a mean reduction to end up with
  // an output of size [N], [] or [] respecitvely, depending on settings. In
  // this function we take the gradient of this output and turn it into a
  // gradient input that can be applied to the CTC loss function gradient by
  // multiplying the incoming gradient with the partial derivative of the
  // reduction.

  const auto &op   = getOp<CtcGradOp>();
  const unsigned N = op.output->tensor(CtcGradOp::getLogProbsGradientOutIndex())
                         ->info.shape()[1];

  if (op.getReductionType() == ReductionType::NoReduction) {

    // Return the incoming gradient as-is as reduction is a no-op.
    return ctcLossGrad;

  } else if (op.getReductionType() == ReductionType::Mean) {

    // A mean reduction is essentially a sum reduction with an subsequent
    // multiplication by 1/totalSamples. The sum reduction's partial
    // derivative is 1 for every loss element, so multiplying the gradient
    // is pointless. The multiplication by 1/totalSamples has a partial
    // derivative of 1/totalSamples, so what we need to do is broadcast
    // the tensor to the right size and multiply by this scalar.

    float totalSamples = static_cast<float>(N);

    // Take into account gradient for mean reduction.
    auto newCtcLossGrad =
        popops::map(graph().getPoplarGraph(),
                    pe::Mul(pe::_1, pe::Const(1.0f / totalSamples)),
                    {ctcLossGrad.getPoplarTensor()},
                    prog,
                    debugContext("divBySamples"));

    // Expand tensor to [N].
    newCtcLossGrad = newCtcLossGrad.expand({0}).broadcast(N, 0);

    // Take into account gradient for division by length.
    newCtcLossGrad =
        popops::map(graph().getPoplarGraph(),
                    pe::Divide(pe::_1,
                               pe::Cast(pe::Max(pe::_2, pe::Const(1)),
                                        newCtcLossGrad.elementType())),
                    {newCtcLossGrad, targetLengths.getPoplarTensor()},
                    prog,
                    debugContext("divByTargetLen"));
    return snap::Tensor{newCtcLossGrad, graph()};

  } else if (op.getReductionType() == ReductionType::Sum) {

    // The partial derivative of a sum reduction with respect to each
    // individual loss element is 1, so applying the chain rule multiplication
    // to the incoming gradient has no effect. We just need to broadcast the
    // scalar into a tensor of shape [N] to get the result we need.

    return snap::Tensor{
        ctcLossGrad.expand({0}).getPoplarTensor().broadcast(N, 0), graph()};

  } else {

    throw error("Unsupported reduction type for Loss {}",
                debugContext().getPathName());
  }
}

namespace {
static OpxCreator<CtcOpx> ctcOpxCreator(Onnx::CustomOperators::Ctc);
static OpxCreator<CtcGradOpx>
    CtcGradOpxCreator(Onnx::CustomGradOperators::CtcGrad);
} // namespace

} // namespace popx
} // namespace popart
