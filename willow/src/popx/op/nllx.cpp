// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <memory>
#include <popart/error.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/op/nll.hpp>
#include <popart/optimizer.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/nllx.hpp>
#include <popart/popx/opxmanager.hpp>

#include <popops/Cast.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Encoding.hpp>
#include <popops/Expr.hpp>
#include <popops/Reduce.hpp>

namespace pe = popops::expr;

namespace popart {
namespace popx {

NllOpx::NllOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<NllOp>(op, Onnx::CustomOperators::Nll);
}

void NllOpx::grow(poplar::program::Sequence &prog) const {
  const NllOp &op = getOp<NllOp>();

  const poplar::Tensor &probs = getInTensor(NllOp::getProbsInIndex());
  const poplar::Tensor &label = getInTensor(NllOp::getLabelInIndex());
  poplar::Tensor probs2D;
  poplar::Tensor label1D;
  poplar::Tensor oneHot;

  flattenAndEncodeOneHot(*this, prog, probs, label, probs2D, label1D, oneHot);

  // oneHot, from a tensor which is sparse with a single 1 per row,
  //           to a tensor which is sparse with a single p per row.
  popops::mapInPlace(graph(),
                     popops::expr::BinaryOpType::MULTIPLY,
                     oneHot,
                     probs2D,
                     prog,
                     debugPrefix("mul"));

  // sum rows, so that just the p corresponding to the label remains
  poplar::Tensor reduction = popops::reduce(graph(),
                                            oneHot,
                                            {1},
                                            {popops::Operation::ADD},
                                            prog,
                                            debugPrefix("reduce"));

  // Create an epsilon value
  poplar::Tensor eps =
      getConst(probs.elementType(), {1}, 1.0e-7, debugPrefix("epsilon"));
  // Add eps (1.0e-7 const value) to reduction make sure it does not have any
  // 0's and log it,
  popops::mapInPlace(graph(),
                     pe::Log(pe::Add(pe::_1, pe::_2)),
                     {reduction, eps},
                     prog,
                     debugPrefix("AddEpsLog"));

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

void NllOpx::flattenAndEncodeOneHot(const Opx &opx,
                                    poplar::program::Sequence &prog,
                                    const poplar::Tensor &probs,
                                    const poplar::Tensor &label,
                                    poplar::Tensor &probs2D,
                                    poplar::Tensor &label1D,
                                    poplar::Tensor &oneHot) {
  // Expect an N-d Probs tensor and (N-1)-d Label tensor.
  // Probs - a tensor of shape [D1, ..., DN, NumClasses]
  // Label - a tensor of shape [D1, ..., DN], where each element is a
  //         class index
  // If N > 2, then the inputs are flattened across all dimenions
  // (except the outer Classes dim in the case of Probs)
  probs2D = probs.flatten(0, probs.rank() - 1);
  label1D = label.flatten();
  // Tensor taking one-hot encoded output must be 2 dimensional
  oneHot = opx.graph().clone(
      probs2D.elementType(), probs2D, opx.debugPrefix("oneHot"));
  popops::encodeOneHot(
      opx.graph(), label1D, oneHot, prog, opx.debugPrefix("nll"));
}

void NllOpx::applyScalingInPlaceForMeanReduction(
    const Opx &opx,
    poplar::Tensor t,
    poplar::program::Sequence &prog,
    bool negate) {
  double totalSamples =
      static_cast<double>(opx.getDevicex()->getReplicationFactor()) *
      static_cast<double>(t.dim(0));
  auto t_totalSamples = opx.getConst(t.elementType(),
                                     {},
                                     negate ? -totalSamples : totalSamples,
                                     opx.debugPrefix("samples"));
  popops::mapInPlace(opx.graph(),
                     popops::expr::BinaryOpType::DIVIDE,
                     t,
                     t_totalSamples,
                     prog,
                     opx.debugPrefix("mean"));
}

void NllOpx::applyScalingInPlaceForMeanReductionWithIgnoreIndex(
    const Opx &opx,
    poplar::Tensor t,
    poplar::Tensor mask,
    poplar::program::Sequence &prog,
    bool negate) {
  // Determine the scale-factor for mean reduction dynamically from the
  // mask.
  // Any sample whose label index is the 'ignore index' should not be
  // counted when scaling the loss/loss grad
  if (mask.rank() == 2 && mask.dim(1) == 1) {
    mask = mask.squeeze({1});
  }
  auto numNonIgnoredSamples =
      popops::reduce(opx.graph(), mask, {0}, {popops::Operation::ADD}, prog);

  double scale = static_cast<double>(opx.getDevicex()->getReplicationFactor());
  if (negate) {
    scale *= -1.0;
  }

  auto repFactor = opx.getConst(
      t.elementType(), {}, scale, opx.debugPrefix("replicationFactor"));

  popops::mapInPlace(opx.graph(),
                     pe::Divide(pe::_1, pe::Mul(pe::_2, pe::_3)),
                     {t, repFactor, numNonIgnoredSamples},
                     prog,
                     opx.debugPrefix("mean"));
}

poplar::Tensor
NllOpx::applyMaskInPlaceForIgnoredIndex(const Opx &opx,
                                        poplar::Tensor t,
                                        poplar::Tensor labels,
                                        int ignoreIndex,
                                        poplar::program::Sequence &prog) {
  // Get the scalar ignoreIndex tensor. If it doens't already
  // exist, create it
  auto ignoreIndexTensor = opx.graph().addConstant(
      labels.elementType(), {}, ignoreIndex, opx.debugPrefix("ignoreIndex"));
  opx.graph().setTileMapping(ignoreIndexTensor, 0);

  // Create the mask
  auto lossMaskBool = popops::map(opx.graph(),
                                  popops::expr::BinaryOpType::NOT_EQUAL,
                                  labels,
                                  ignoreIndexTensor,
                                  prog,
                                  opx.debugPrefix("notEqual"));
  auto lossMask     = popops::cast(opx.graph(),
                               lossMaskBool,
                               t.elementType(),
                               prog,
                               opx.debugPrefix("cast"));

  // Expand, if required, for valid broadcasting of mul
  if (t.rank() == 2) {
    lossMask = lossMask.expand({1});
  }

  // Apply the mask
  popops::mapInPlace(opx.graph(),
                     popops::expr::BinaryOpType::MULTIPLY,
                     t,
                     lossMask,
                     prog,
                     opx.debugPrefix("masked"));

  return lossMask;
}

void NllOpx::handleLossOutNotReducedToScalar(const Opx &opx,
                                             poplar::Tensor &reduction,
                                             const poplar::Tensor &label,
                                             poplar::Tensor &label1D,
                                             poplar::program::Sequence &prog) {
  popops::mapInPlace(opx.graph(),
                     popops::expr::UnaryOpType::NEGATE,
                     reduction,
                     prog,
                     opx.debugPrefix("neg"));
  // One loss per sample, so the output is reshaped to match label input shape
  reduction = reduction.reshape(label.shape());

  opx.setOutTensor(0, reduction);
}

void NllOpx::handleLossOutReducedToScalar(const Opx &opx,
                                          bool hasIgnoreIndex,
                                          int64_t ignoreIndex,
                                          bool meanReduce,
                                          poplar::Tensor &reduction,
                                          poplar::Tensor &label1D,
                                          poplar::program::Sequence &prog,
                                          const OutIndex outIdx) {
  double scale = 1.0;

  if (meanReduce) {
    if (hasIgnoreIndex) {
      auto lossMask = applyMaskInPlaceForIgnoredIndex(
          opx, reduction, label1D, ignoreIndex, prog);
      applyScalingInPlaceForMeanReductionWithIgnoreIndex(
          opx, reduction, lossMask, prog);
      // Leave scale as 1.0 as already scaled
    } else {
      double totalSamples =
          static_cast<double>(opx.getDevicex()->getReplicationFactor()) *
          static_cast<double>(reduction.dim(0));
      scale = 1.0 / totalSamples;
    }
  }

  // Scale (possibly) and negate (-scale)
  auto t_scale =
      opx.getConst(poplar::FLOAT, {}, -scale, opx.debugPrefix("scale"));

  auto scalar = popops::reduce(opx.graph(),
                               reduction,
                               {0},
                               {popops::Operation::ADD, false, t_scale},
                               prog,
                               opx.debugPrefix("toScalar"));
  opx.setOutTensor(outIdx, scalar);
}

NllGradOpx::NllGradOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<NllGradOp>(op, Onnx::CustomGradOperators::NllGrad);
}

// loss         = -ln (p_l), where p_l is the probability at "label" so
//
//                 0     if i != l
// d_loss / d_p = -1/p_i if i == l
//                 ...............

void NllGradOpx::grow(poplar::program::Sequence &prog) const {
  const NllGradOp &gradOp     = getOp<NllGradOp>();
  const poplar::Tensor &probs = getInTensor(NllGradOp::getProbsInIndex());
  const poplar::Tensor &label = getInTensor(NllGradOp::getLabelInIndex());
  poplar::Tensor gradIn       = getInTensor(NllGradOp::getGradInIndex());

  // As for NllOpx, flatten outer dimenstions if rank(probs) > 2
  auto probs2D = probs.flatten(0, probs.rank() - 1);
  auto label1D = label.flatten();

  // inverse probabilities, we take max(eps, p) to make division safe
  float eps = 1e-10f;
  auto smallConst =
      graph().addConstant(probs.elementType(), {1}, eps, debugPrefix("eps"));
  graph().setTileMapping(smallConst, 0);
  auto safeProbs = popops::map(graph(),
                               popops::expr::BinaryOpType::MAXIMUM,
                               smallConst,
                               probs2D,
                               prog,
                               debugPrefix("max"));

  // oneHot: initialised to be 1 at position "label", 0 elsewhere.
  auto oneHot =
      graph().clone(probs2D.elementType(), probs2D, debugPrefix("oneHot"));

  popops::encodeOneHot(graph(), label1D, oneHot, prog, debugPrefix("nll"));

  // oneHot: becomes -1 at position "label", 0 elsewhere.
  // oneHot: set to -1/p at position "label", 0 elsewhere.
  popops::mapInPlace(graph(),
                     pe::Divide(pe::Neg(pe::_1), pe::_2),
                     {oneHot, safeProbs},
                     prog,
                     debugPrefix("NegDiv"));

  // Apply mask before reduction, so that ignored class doesn't
  // contribute to the loss gradient
  if (gradOp.hasIgnoreIndex()) {
    auto lossMask = NllOpx::applyMaskInPlaceForIgnoredIndex(
        *this, oneHot, label1D, gradOp.getIgnoreIndex(), prog);

    if (gradOp.getReductionType() == ReductionType::Mean) {
      NllOpx::applyScalingInPlaceForMeanReductionWithIgnoreIndex(
          *this, oneHot, lossMask, prog);
    }
  } else {
    if (gradOp.getReductionType() == ReductionType::Mean) {
      NllOpx::applyScalingInPlaceForMeanReduction(*this, oneHot, prog);
    }
  }

  // Output is reshaped to match probs input shape
  oneHot = oneHot.reshape(probs.shape());

  // To ensure gradIn has a broadcastable shape, add extra singleton dimensions
  for (unsigned dim = 0; dim < oneHot.rank(); dim++) {
    if (dim > gradIn.rank() - 1) {
      gradIn = gradIn.expand({dim});
    }
  }
  popops::mapInPlace(graph(),
                     pe::Mul(pe::_1, pe::_2),
                     {oneHot, gradIn},
                     prog,
                     debugPrefix("scaledGradIn"));

  setOutTensor(0, oneHot);
}

namespace {
static OpxCreator<NllOpx> nllOpxCreator(Onnx::CustomOperators::Nll);
static OpxCreator<NllGradOpx>
    nllGradOpxCreator(Onnx::CustomGradOperators::NllGrad);
} // namespace

} // namespace popx
} // namespace popart
