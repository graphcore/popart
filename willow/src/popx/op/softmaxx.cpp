// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <cstddef>
#include <snap/Graph.hpp>
#include <snap/Program.hpp>
#include <snap/Tensor.hpp>
#include <snap/popops/ElementWise.hpp>
#include <string>
#include <vector>
#include <poplar/Tensor.hpp>
#include <popnn/NonLinearity.hpp>
#include <popnn/NonLinearityDef.hpp>
#include <popops/Expr.hpp>
#include <popops/ExprOp.hpp>
#include <popops/OperationDef.hpp>
#include <popops/Reduce.hpp>
#include <popart/error.hpp>
#include <popart/ir.hpp>
#include <popart/op/softmax.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/nllx.hpp>
#include <popart/popx/op/softmaxx.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/logging.hpp"
#include "popart/op.hpp"
#include "popart/operatoridentifier.hpp"
#include "popart/operators.hpp"
#include "popart/popx/debugcontextx.hpp"
#include "popart/popx/op/elementwisex.hpp"
#include "popart/popx/popopx.hpp"
#include "popart/sessionoptions.hpp"
#include "popart/tensorinfo.hpp"

namespace pe = popops::expr;

namespace popart {
namespace popx {

namespace {
template <typename T> T *getAs(Op *op) {
  auto x = dynamic_cast<T *>(op);
  if (!x) {
    throw error("Failed to cast {} in Softmaxx", op->str());
  }
  return x;
}
} // namespace

SoftmaxInplaceOpx::SoftmaxInplaceOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryInplaceOpx(
          op,
          devicex,
          SoftmaxComputex::get(
              getAs<SoftmaxInplaceOp>(op)->getAxis(),
              devicex->ir().getSessionOptions().enableNonStableSoftmax,
              op->inInfo(SoftmaxInplaceOp::getInIndex()).shape_szt())) {}

SoftmaxOpx::SoftmaxOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryOutplaceOpx(
          op,
          devicex,
          SoftmaxComputex::get(
              getAs<SoftmaxOp>(op)->getAxis(),
              devicex->ir().getSessionOptions().enableNonStableSoftmax,
              op->inInfo(SoftmaxOp::getInIndex()).shape_szt())) {}

snap::Tensor SoftmaxComputex::outplace(snap::program::Sequence &p,
                                       snap::Graph &g,
                                       const snap::Tensor &t,
                                       const poplar::DebugNameAndId &dnai,
                                       const std::string &s) const {
  auto outTensor = cloneNcopy(p, g, t, dnai);
  inplace(p, g, outTensor, dnai, s);
  return outTensor;
}

void SoftmaxComputex::inplace(snap::program::Sequence &p,
                              snap::Graph &g,
                              const snap::Tensor &tIn,
                              const poplar::DebugNameAndId &dnai,
                              const std::string &dbs) const {

  auto input = coerceTo2D(tIn, axis);

  // By default use stable softmax (prevent overflow by subtracting max
  // input value from input tensor before computing the exponentials).
  // Optionally override.
  popnn::NonLinearityType nlType;
  if (enableNonStable) {
    nlType = popnn::NonLinearityType::SOFTMAX;
  } else {
    nlType = popnn::NonLinearityType::SOFTMAX_STABLE;
  }

  popnn::nonLinearityInPlace(g.getPoplarGraph(),
                             nlType,
                             input.getPoplarTensor(),
                             p.getPoplarSequence(),
                             {dnai, dbs});
}

snap::Tensor SoftmaxComputex::reshape(const snap::Tensor &t) const {
  return t.reshape(outShape);
}

SoftmaxGradOpx::SoftmaxGradOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryOpx(op, devicex) {
  verifyOp<SoftmaxGradOp>(op, Onnx::GradOperators::SoftmaxGrad);
}

SoftmaxGradDirectOpx::SoftmaxGradDirectOpx(Op *op, Devicex *devicex)
    : PopOpx(op, devicex) {
  verifyOp<SoftmaxGradDirectOp>(op,
                                Onnx::CustomGradOperators::SoftmaxGradDirect);
}

NlllWithSoftmaxGradDirectOpx::NlllWithSoftmaxGradDirectOpx(Op *op,
                                                           Devicex *devicex)
    : PopOpx(op, devicex) {
  verifyOp<NlllWithSoftmaxGradDirectOp>(
      op, Onnx::CustomGradOperators::NlllWithSoftmaxGradDirect);
}

// The maths for SoftmaxGradDirect:
//   loss = -ln(p_j), where j is the true class
//   d(loss)/d(p_i) = 0, d(loss)/d(p_j) = -1/p_j
//   p_j = exp(v_j) / S
//   where S = sum_{all indices k} [ exp(v_k) ]
//   By the quotient rule:
//   d(p_j)/d(v_i)  = (0 - exp(v_j).exp(v_i)) / S^2
//                  = -p_i.p_j
//   d(p_j)/d(v_j)  = (exp(v_j).S - exp(v_j).exp(v_j)) / S^2
//                  = p_j - p_i.p_j
//   Then, using the chain rule,
//   d(loss)/d(v_i) = p_i
//   d(loss)/d(v_j) = p_j - 1

void SoftmaxGradDirectOpx::grow(snap::program::Sequence &prog) const {
  SoftmaxGradDirectOp &op = getOp<SoftmaxGradDirectOp>();
  const snap::Tensor &probs =
      getInTensor(SoftmaxGradDirectOp::getProbsInIndex());
  const snap::Tensor &label =
      getInTensor(SoftmaxGradDirectOp::getLabelInIndex());
  snap::Tensor gradIn = getInTensor(SoftmaxGradDirectOp::getGradProbsInIndex());

  snap::Tensor probs2D;
  snap::Tensor label1D;
  snap::Tensor oneHot;

  NllOpx::flattenAndEncodeOneHot(
      *this, prog, probs, label, probs2D, label1D, oneHot);

  // -1 at position "label", 0 elsewhere.
  // p - 1 at position "label" label, p elsewhere.
  snap::popops::mapInPlace(graph(),
                           pe::Add(pe::Neg(pe::_1), pe::_2),
                           {oneHot, probs2D},
                           prog,
                           debugContext("negsub"));

  // Output is reshaped to match probs input shape
  oneHot = oneHot.reshape(probs.shape());

  NllOpx::handleLossGradScaling(*this,
                                op.hasIgnoreIndex(),
                                op.hasIgnoreIndex() ? op.getIgnoreIndex() : 0,
                                op.getReductionType() == ReductionType::Mean,
                                oneHot,
                                gradIn,
                                label1D,
                                prog);

  setOutTensor(0, oneHot);
}

void SoftmaxGradOpx::grow(snap::program::Sequence &prog) const {
  const auto axis = getOp<SoftmaxGradOp>().getAxis();

  // Note: Implementation for SOFTMAX and SOFTMAX_STABLE gradient
  //       are the same.

  auto outActs   = getInTensor(SoftmaxGradOp::getProbsInIndex());
  auto outActs2D = EwuComputex::coerceTo2D(outActs, axis);
  auto outGrad   = getInTensor(SoftmaxGradOp::getGradProbsInIndex())
                     .getPoplarTensor()
                     .reshape(outActs2D.shape());
  auto outTensor = popnn::nonLinearityInputGradient(
      graph().getPoplarGraph(),                // graph,
      popnn::NonLinearityType::SOFTMAX_STABLE, // nonLinearityType
      outActs2D.getPoplarTensor(),             // out,
      outGrad,                                 // outGradient,
      prog.getPoplarSequence(),                // prog,
      debugContext("SoftmaxGrad")              // debugContext
  );

  setOutTensor(0, snap::Tensor{outTensor.reshape(outActs.shape()), graph()});
}

void NlllWithSoftmaxGradDirectOpx::grow(snap::program::Sequence &prog) const {
  NlllWithSoftmaxGradDirectOp &op = getOp<NlllWithSoftmaxGradDirectOp>();

  const snap::Tensor &probs =
      getInTensor(NlllWithSoftmaxGradDirectOp::getProbsInIndex());
  const snap::Tensor &label =
      getInTensor(NlllWithSoftmaxGradDirectOp::getLabelInIndex());
  snap::Tensor gradIn =
      getInTensor(NlllWithSoftmaxGradDirectOp::getGradProbsInIndex());
  snap::Tensor probs2D;
  snap::Tensor label1D;
  snap::Tensor oneHot;

  NllOpx::flattenAndEncodeOneHot(
      *this, prog, probs, label, probs2D, label1D, oneHot);

  // oneHotProbs, from a tensor which is sparse with a single 1 per row,
  //              to a tensor which is sparse with a single p per row.
  auto oneHotProbs = snap::popops::map(graph(),
                                       popops::expr::BinaryOpType::MULTIPLY,
                                       oneHot,
                                       probs2D,
                                       prog,
                                       debugContext("mul"));

  // Now compute the SoftmaxGrad:

  // TODO: T8303
  // -1 at position "label", 0 elsewhere.
  // p - 1 at position "label" label, p elsewhere.
  snap::popops::mapInPlace(graph(),
                           pe::Add(pe::Neg(pe::_1), pe::_2),
                           {oneHot, probs2D},
                           prog,
                           debugContext("NegSub"));

  // Output is reshaped to match probs input shape
  oneHot = oneHot.reshape(probs.shape());

  NllOpx::handleLossGradScaling(*this,
                                op.hasIgnoreIndex(),
                                op.hasIgnoreIndex() ? op.getIgnoreIndex() : 0,
                                op.getReductionType() == ReductionType::Mean,
                                oneHot,
                                gradIn,
                                label1D,
                                prog);

  setOutTensor(op.getGradOutIndex(), oneHot);

  // Now compute the rest of the nll loss from the same one-hot encoded tensor:

  // sum rows, so that just the p corresponding to the label remains
  snap::Tensor reduction =
      snap::Tensor{popops::reduce(graph().getPoplarGraph(),
                                  oneHotProbs.getPoplarTensor(),
                                  {1},
                                  {popops::Operation::ADD},
                                  prog.getPoplarSequence(),
                                  debugContext("add")),
                   graph()};

  // Create an epsilon value
  snap::Tensor eps = getConst(probs.elementType(), {1}, 1.0e-7, "epsilon");
  // Add eps to reduction to make sure it does not have any 0's and log it,
  snap::popops::mapInPlace(graph(),
                           pe::Log(pe::Add(pe::_1, pe::_2)),
                           {reduction, eps},
                           prog,
                           debugContext("LogEpsMul"));

  // TODO: T8305, re-use the mask created above
  if (op.hasIgnoreIndex()) {
    auto lossMask = NllOpx::applyMaskInPlaceForIgnoredIndex(
        *this, reduction, label1D, op.getIgnoreIndex(), prog);
  }

  if (op.getReductionType() == ReductionType::NoReduction) {
    NllOpx::handleLossOutNotReducedToScalar(
        *this, reduction, label, label1D, prog);
  } else {
    NllOpx::handleLossOutReducedToScalar(
        *this,
        op.hasIgnoreIndex(),
        op.hasIgnoreIndex() ? op.getIgnoreIndex() : 0,
        op.getReductionType() == ReductionType::Mean,
        reduction,
        label1D,
        prog,
        op.getLossOutIndex());
  }
}

namespace {
OpxCreator<SoftmaxOpx> softmaxOpxCreator({Onnx::Operators::Softmax_1,
                                          Onnx::Operators::Softmax_11});
OpxCreator<SoftmaxGradOpx>
    softmaxGradOpxCreator(Onnx::GradOperators::SoftmaxGrad);
OpxCreator<SoftmaxGradDirectOpx>
    softmaxGradDirectOpxCreator(Onnx::CustomGradOperators::SoftmaxGradDirect);
OpxCreator<NlllWithSoftmaxGradDirectOpx> nlllWithSoftmaxGradDirectOpxCreator(
    Onnx::CustomGradOperators::NlllWithSoftmaxGradDirect);
OpxCreator<SoftmaxInplaceOpx>
    softmaxxInplaceOpxCreator(Onnx::CustomOperators::SoftmaxInplace);

} // namespace

} // namespace popx
} // namespace popart
