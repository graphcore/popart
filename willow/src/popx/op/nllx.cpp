// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include "popart/logging.hpp"
#include <memory>
#include <popart/error.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/op/nll.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/nllx.hpp>
#include <popart/popx/opxmanager.hpp>

#include <poplar/Tensor.hpp>
#include <popops/Cast.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Encoding.hpp>
#include <popops/Expr.hpp>
#include <popops/Reduce.hpp>

namespace pe = popops::expr;

namespace popart {
namespace popx {

NllOpx::NllOpx(Op *op, Devicex *devicex) : PopOpx(op, devicex) {
  verifyOp<NllOp>(op, Onnx::CustomOperators::Nll);
}

void NllOpx::grow(snap::program::Sequence &prog) const {
  const NllOp &op = getOp<NllOp>();

  const snap::Tensor &probs = getInTensor(NllOp::getProbsInIndex());
  const snap::Tensor &label = getInTensor(NllOp::getLabelInIndex());
  snap::Tensor probs2D;
  snap::Tensor label1D;
  snap::Tensor oneHot;

  flattenAndEncodeOneHot(*this, prog, probs, label, probs2D, label1D, oneHot);

  // oneHot, from a tensor which is sparse with a single 1 per row,
  //           to a tensor which is sparse with a single p per row.
  popops::mapInPlace(graph().getPoplarGraph(),
                     popops::expr::BinaryOpType::MULTIPLY,
                     oneHot.getPoplarTensor(),
                     probs2D.getPoplarTensor(),
                     prog.getPoplarSequence(),
                     debugContext("mul"));

  // sum rows, so that just the p corresponding to the label remains
  snap::Tensor reduction = snap::Tensor{popops::reduce(graph().getPoplarGraph(),
                                                       oneHot.getPoplarTensor(),
                                                       {1},
                                                       {popops::Operation::ADD},
                                                       prog.getPoplarSequence(),
                                                       debugContext("reduce")),
                                        graph()};

  // Create an epsilon value
  double eps_f = 1.0e-7;

  // if using fp16, increase the eps to avoid underfloat
  if (probs.elementType() == poplar::HALF)
    eps_f = 6.104e-05;

  snap::Tensor eps = getConst(probs.elementType(), {1}, eps_f, "epsilon");

  if (!op.inputIsLogProbability()) {
    // Take max of prob and eps to reduction make sure it does not have any
    // 0's and log it,
    popops::mapInPlace(graph().getPoplarGraph(),
                       pe::Log(pe::Max(pe::_1, pe::_2)),
                       {reduction.getPoplarTensor(), eps.getPoplarTensor()},
                       prog.getPoplarSequence(),
                       debugContext("LogMax"));
  }

  if (op.hasIgnoreIndex()) {
    auto lossMask = applyMaskInPlaceForIgnoredIndex(
        *this, reduction, label1D, op.getIgnoreIndex(), prog);
  }

  if (op.getReductionType() == ReductionType::NoReduction) {
    handleLossOutNotReducedToScalar(*this, reduction, label, label1D, prog);
  } else {
    handleLossOutReducedToScalar(*this,
                                 op.hasIgnoreIndex(),
                                 op.hasIgnoreIndex() ? op.getIgnoreIndex() : 0,
                                 op.getReductionType() == ReductionType::Mean,
                                 reduction,
                                 label1D,
                                 prog,
                                 op.getOutIndex());
  }
}

void NllOpx::flattenAndEncodeOneHot(const PopOpx &opx,
                                    snap::program::Sequence &prog,
                                    const snap::Tensor &probs,
                                    const snap::Tensor &label,
                                    snap::Tensor &probs2D,
                                    snap::Tensor &label1D,
                                    snap::Tensor &oneHot) {
  // Expect an N-d Probs tensor and (N-1)-d Label tensor.
  // Probs - a tensor of shape [D1, ..., DN, NumClasses]
  // Label - a tensor of shape [D1, ..., DN], where each element is a
  //         class index
  // If N > 2, then the inputs are flattened across all dimensions
  // (except the outer Classes dim in the case of Probs)
  probs2D = snap::Tensor{probs.flatten(0, probs.rank() - 1).getPoplarTensor(),
                         opx.graph()};
  label1D = snap::Tensor{label.flatten().getPoplarTensor(), opx.graph()};
  // Tensor taking one-hot encoded output must be 2 dimensional
  oneHot = snap::Tensor{
      opx.graph().getPoplarGraph().clone(probs2D.elementType(),
                                         probs2D.getPoplarTensor(),
                                         opx.debugContext("oneHot")),
      opx.graph()};
  popops::encodeOneHot(opx.graph().getPoplarGraph(),
                       label1D.getPoplarTensor(),
                       oneHot.getPoplarTensor(),
                       prog.getPoplarSequence(),
                       opx.debugContext("nll"));
}

void NllOpx::applyScalingInPlaceForMeanReduction(
    const PopOpx &opx,
    snap::Tensor t,
    snap::Tensor scale,
    snap::program::Sequence &prog) {
  double totalSamples = static_cast<double>(t.dim(0));

  auto combined_scale = popops::div(opx.graph().getPoplarGraph(),
                                    scale.getPoplarTensor(),
                                    totalSamples,
                                    prog.getPoplarSequence(),
                                    opx.debugContext("combinedLossScale"));

  // Note: if combined_scale is fp32 and t is fp16, the downcast is handled
  // here by poplar
  popops::mulInPlace(opx.graph().getPoplarGraph(),
                     t.getPoplarTensor(),
                     combined_scale,
                     prog.getPoplarSequence(),
                     opx.debugContext("mean"));
}

void NllOpx::applyScalingInPlaceForMeanReductionWithIgnoreIndex(
    const PopOpx &opx,
    snap::Tensor t,
    snap::Tensor scale,
    snap::Tensor mask,
    snap::program::Sequence &prog) {
  // Determine the scale-factor for mean reduction dynamically from the
  // mask.
  // Any sample whose label index is the 'ignore index' should not be
  // counted when scaling the loss/loss grad
  auto numNonIgnoredSamples =
      popops::reduce(opx.graph().getPoplarGraph(),
                     mask.flatten().getPoplarTensor(),
                     {0},
                     {popops::Operation::ADD},
                     prog.getPoplarSequence(),
                     opx.debugContext("numNonIgnoredSamples"));

  // If the numNonIgnoredSamples is equal to zero, we have ignored all label
  // data, in this case return zero loss. Do this by taking
  // min(numIgnoredSamples, 1) and letting the result be 0 / 1 (where scale = 0
  // due to the ignored labels). See T36441
  auto min_1 = opx.graph().getPoplarGraph().addConstant(
      numNonIgnoredSamples.elementType(), {}, 1, opx.debugContext("const_1"));
  opx.graph().getPoplarGraph().setTileMapping(min_1, 0);
  popops::maxInPlace(opx.graph().getPoplarGraph(),
                     numNonIgnoredSamples,
                     min_1,
                     prog.getPoplarSequence(),
                     opx.debugContext("numNonIgnoredSamples_min"));

  // popops::div requires inputs of the same data type. We support the mixed
  // case where gradIn is fp32 but the mask tensor is fp32. So here we upcast
  // if required
  if (numNonIgnoredSamples.elementType() !=
      scale.getPoplarTensor().elementType()) {
    numNonIgnoredSamples = popops::cast(opx.graph().getPoplarGraph(),
                                        numNonIgnoredSamples,
                                        scale.getPoplarTensor().elementType(),
                                        prog.getPoplarSequence(),
                                        opx.debugContext("cast"));
  }

  auto combined_scale = popops::div(opx.graph().getPoplarGraph(),
                                    scale.getPoplarTensor(),
                                    numNonIgnoredSamples,
                                    prog.getPoplarSequence(),
                                    opx.debugContext("combinedLossScale"));

  // Note: if combined_scale is fp32 and t is fp16, the downcast is handled
  // here by poplar
  popops::mulInPlace(opx.graph().getPoplarGraph(),
                     t.getPoplarTensor(),
                     combined_scale,
                     prog.getPoplarSequence(),
                     opx.debugContext("mean"));
}

snap::Tensor
NllOpx::applyMaskInPlaceForIgnoredIndex(const PopOpx &opx,
                                        snap::Tensor t,
                                        snap::Tensor labels,
                                        int ignoreIndex,
                                        snap::program::Sequence &prog) {
  // Get the scalar ignoreIndex tensor. If it doesn't already
  // exist, create it
  auto ignoreIndexTensor = opx.graph().getPoplarGraph().addConstant(
      labels.elementType(), {}, ignoreIndex, opx.debugContext("ignoreIndex"));
  opx.graph().getPoplarGraph().setTileMapping(ignoreIndexTensor, 0);

  // Create the mask
  auto lossMaskBool = popops::map(opx.graph().getPoplarGraph(),
                                  popops::expr::BinaryOpType::NOT_EQUAL,
                                  labels.getPoplarTensor(),
                                  ignoreIndexTensor,
                                  prog.getPoplarSequence(),
                                  opx.debugContext("notEqual"));
  auto lossMask     = popops::cast(opx.graph().getPoplarGraph(),
                               lossMaskBool,
                               t.elementType(),
                               prog.getPoplarSequence(),
                               opx.debugContext("cast"));

  if (t.rank() != lossMask.rank()) {
    // If required, broadcast lossMask on the final dimension.
    auto t_shape          = t.shape();
    t_shape[t.rank() - 1] = 1;
    lossMask              = lossMask.reshape(t_shape);
  }

  // Apply the mask
  popops::mapInPlace(opx.graph().getPoplarGraph(),
                     popops::expr::BinaryOpType::MULTIPLY,
                     t.getPoplarTensor(),
                     lossMask,
                     prog.getPoplarSequence(),
                     opx.debugContext("masked"));

  return snap::Tensor{lossMask, opx.graph()};
}

void NllOpx::handleLossOutNotReducedToScalar(const PopOpx &opx,
                                             snap::Tensor &reduction,
                                             const snap::Tensor &label,
                                             snap::Tensor &label1D,
                                             snap::program::Sequence &prog) {
  popops::mapInPlace(opx.graph().getPoplarGraph(),
                     popops::expr::UnaryOpType::NEGATE,
                     reduction.getPoplarTensor(),
                     prog.getPoplarSequence(),
                     opx.debugContext("neg"));
  // One loss per sample, so the output is reshaped to match label input shape
  reduction = reduction.reshape(label.shape());

  opx.setOutTensor(0, reduction);
}

void NllOpx::handleLossOutReducedToScalar(const PopOpx &opx,
                                          bool hasIgnoreIndex,
                                          int64_t ignoreIndex,
                                          bool meanReduce,
                                          snap::Tensor &reduction,
                                          snap::Tensor &label1D,
                                          snap::program::Sequence &prog,
                                          const OutIndex outIdx) {
  double scale = 1.0;

  if (meanReduce) {
    if (hasIgnoreIndex) {
      auto lossMask = applyMaskInPlaceForIgnoredIndex(
          opx, reduction, label1D, static_cast<int>(ignoreIndex), prog);

      auto scaleT = opx.getConst(reduction.elementType(), {}, 1.0, "One");

      applyScalingInPlaceForMeanReductionWithIgnoreIndex(
          opx, reduction, scaleT, lossMask, prog);
      // Leave scale as 1.0 as already scaled
    } else {
      double totalSamples = static_cast<double>(reduction.dim(0));
      scale               = 1.0 / totalSamples;
    }
  }

  // Scale (possibly) and negate (-scale)
  auto t_scale =
      opx.getConst(poplar::FLOAT, {}, -scale, "scale").getPoplarTensor();

  auto scalar = popops::reduce(opx.graph().getPoplarGraph(),
                               reduction.getPoplarTensor(),
                               {0},
                               {popops::Operation::ADD, false, t_scale},
                               prog.getPoplarSequence(),
                               opx.debugContext("toScalar"));
  opx.setOutTensor(outIdx, snap::Tensor{scalar, opx.graph()});
}

void NllOpx::handleLossGradScaling(const PopOpx &opx,
                                   bool hasIgnoreIndex,
                                   int64_t ignoreIndex,
                                   bool meanReduce,
                                   snap::Tensor &oneHot,
                                   snap::Tensor &gradIn,
                                   snap::Tensor &label1D,
                                   snap::program::Sequence &prog) {
  // To ensure gradIn has a broadcastable shape, add extra singleton
  // dimensions
  for (unsigned dim = 0; dim < oneHot.rank(); dim++) {
    if (dim > gradIn.rank() - 1) {
      gradIn = gradIn.expand({dim});
    }
  }

  // Apply mask before reduction, so that ignored class doesn't
  // contribute to the loss gradient
  if (hasIgnoreIndex) {
    auto lossMask = NllOpx::applyMaskInPlaceForIgnoredIndex(
        opx, oneHot, label1D, ignoreIndex, prog);

    if (meanReduce) {
      NllOpx::applyScalingInPlaceForMeanReductionWithIgnoreIndex(
          opx, oneHot, gradIn, lossMask, prog);
    }
  } else {
    if (meanReduce) {
      NllOpx::applyScalingInPlaceForMeanReduction(opx, oneHot, gradIn, prog);
    }
  }

  if (!meanReduce) {
    popops::mulInPlace(opx.graph().getPoplarGraph(),
                       oneHot.getPoplarTensor(),
                       gradIn.getPoplarTensor(),
                       prog.getPoplarSequence(),
                       opx.debugContext("scaledGradIn"));
  }
}

NllGradOpx::NllGradOpx(Op *op, Devicex *devicex) : PopOpx(op, devicex) {
  verifyOp<NllGradOp>(op, Onnx::CustomGradOperators::NllGrad);
}

// NllGrad depends on whether the input contains log-probabilities:
//
// 1) inputIsLogProbability == false (default)
//    loss         = -ln (p_l), where p_l is the probability at "label" so
//
//                     0     if i != l
//    d_loss / d_p = -1/p_i if i == l
//
// 2) inputIsLogProbability == true (pytorch convention)
//    loss         = -p_l, defined as above
//
//                     0   if i != l
//    d_loss / d_p = -p_i if i == l

void NllGradOpx::grow(snap::program::Sequence &prog) const {
  const NllGradOp &gradOp   = getOp<NllGradOp>();
  const snap::Tensor &probs = getInTensor(NllGradOp::getProbsInIndex());
  const snap::Tensor &label = getInTensor(NllGradOp::getLabelInIndex());
  snap::Tensor gradIn       = getInTensor(NllGradOp::getGradInIndex());

  // As for NllOpx, flatten outer dimensions if rank(probs) > 2
  auto probs2D = snap::Tensor{
      probs.flatten(0, probs.rank() - 1).getPoplarTensor(), graph()};
  auto label1D = snap::Tensor{label.flatten().getPoplarTensor(), graph()};

  // inverse probabilities, we take max(eps, p) to make division safe
  float eps = 1e-10f;

  // If working with float16 increase the eps to avoid underfloat
  // Note that because of the division we would ideally use 1/65500
  // but this will underflow.
  if (probs.elementType() == poplar::HALF) {
    eps = 6.104e-05f;
  }

  auto smallConst = graph().getPoplarGraph().addConstant(
      probs.elementType(), {1}, eps, debugContext("eps"));
  graph().getPoplarGraph().setTileMapping(smallConst, 0);
  auto safeProbs = popops::map(graph().getPoplarGraph(),
                               popops::expr::BinaryOpType::MAXIMUM,
                               smallConst,
                               probs2D.getPoplarTensor(),
                               prog.getPoplarSequence(),
                               debugContext("max"));

  // oneHot: initialised to be 1 at position "label", 0 elsewhere.
  auto oneHot =
      snap::Tensor{graph().getPoplarGraph().clone(probs2D.elementType(),
                                                  probs2D.getPoplarTensor(),
                                                  debugContext("oneHot")),
                   graph()};

  popops::encodeOneHot(graph().getPoplarGraph(),
                       label1D.getPoplarTensor(),
                       oneHot.getPoplarTensor(),
                       prog.getPoplarSequence(),
                       debugContext("nll"));

  if (gradOp.inputIsLogProbability()) {
    // oneHot: becomes -1 at position "label", 0 elsewhere.
    popops::mapInPlace(graph().getPoplarGraph(),
                       pe::UnaryOpType::NEGATE,
                       oneHot.getPoplarTensor(),
                       prog.getPoplarSequence(),
                       debugContext("negOneHot"));
  } else {
    // oneHot: set to -1/p at position "label", 0 elsewhere.
    popops::mapInPlace(graph().getPoplarGraph(),
                       pe::Divide(pe::Neg(pe::_1), pe::_2),
                       {oneHot.getPoplarTensor(), safeProbs},
                       prog.getPoplarSequence(),
                       debugContext("NegDiv"));
  }

  // Output is reshaped to match probs input shape
  oneHot = oneHot.reshape(probs.shape());

  NllOpx::handleLossGradScaling(
      *this,
      gradOp.hasIgnoreIndex(),
      gradOp.hasIgnoreIndex() ? gradOp.getIgnoreIndex() : 0,
      gradOp.getReductionType() == ReductionType::Mean,
      oneHot,
      gradIn,
      label1D,
      prog);

  setOutTensor(0, oneHot);
}

namespace {
static OpxCreator<NllOpx> nllOpxCreator(Onnx::CustomOperators::Nll);
static OpxCreator<NllGradOpx>
    nllGradOpxCreator(Onnx::CustomGradOperators::NllGrad);
} // namespace

} // namespace popx
} // namespace popart
