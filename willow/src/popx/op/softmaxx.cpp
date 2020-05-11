// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <iterator>
#include <memory>
#include <popart/error.hpp>
#include <popart/ir.hpp>
#include <popart/op/nll.hpp>
#include <popart/op/softmax.hpp>
#include <popart/optimizer.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/nllx.hpp>
#include <popart/popx/op/softmaxx.hpp>
#include <popart/popx/opxmanager.hpp>

#include <popnn/NonLinearity.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Encoding.hpp>
#include <popops/Expr.hpp>
#include <popops/Reduce.hpp>

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

poplar::Tensor SoftmaxComputex::outplace(poplar::program::Sequence &p,
                                         poplar::Graph &g,
                                         const poplar::Tensor &t,
                                         const std::string &s) const {
  auto outTensor = cloneNcopy(p, g, t);
  inplace(p, g, outTensor, s);
  return outTensor;
}

namespace {
poplar::Tensor coerceTo2D(const poplar::Tensor &t, int64_t axis) {

  const auto in_shape = t.shape();
  auto k              = in_shape.begin();
  std::advance(k, axis);

  auto n = std::accumulate(
      in_shape.begin(), k, std::size_t{1}, std::multiplies<std::size_t>());
  auto d = std::accumulate(
      k, in_shape.end(), std::size_t{1}, std::multiplies<std::size_t>());
  return t.reshape({n, d});
}
} // namespace

void SoftmaxComputex::inplace(poplar::program::Sequence &p,
                              poplar::Graph &g,
                              const poplar::Tensor &tIn,
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

  popnn::nonLinearityInPlace(g, nlType, input, p, dbs);
}

poplar::Tensor SoftmaxComputex::reshape(const poplar::Tensor &t) const {
  return t.reshape(outShape);
}

SoftmaxGradOpx::SoftmaxGradOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryOpx(op, devicex) {
  verifyOp<SoftmaxGradOp>(op, Onnx::GradOperators::SoftmaxGrad);
}

SoftmaxGradDirectOpx::SoftmaxGradDirectOpx(Op *op, Devicex *devicex)
    : Opx(op, devicex) {
  verifyOp<SoftmaxGradDirectOp>(op,
                                Onnx::CustomGradOperators::SoftmaxGradDirect);
}

NlllWithSoftmaxGradDirectOpx::NlllWithSoftmaxGradDirectOpx(Op *op,
                                                           Devicex *devicex)
    : Opx(op, devicex) {
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

void SoftmaxGradDirectOpx::grow(poplar::program::Sequence &prog) const {
  SoftmaxGradDirectOp &sfmgd  = getOp<SoftmaxGradDirectOp>();
  const poplar::Tensor &probs = getInTensor(NllLoss::getProbsInIndex());
  const poplar::Tensor &label = getInTensor(NllLoss::getLabelInIndex());

  // As for NllOpx, flatten outer dimensions if rank(probs) > 2
  auto probs2D = probs.flatten(0, probs.rank() - 1);
  auto label1D = label.flatten();

  // 1 at position "label", 0 elsewhere.
  auto oneHot =
      graph().clone(probs2D.elementType(), probs2D, debugPrefix("oneHot"));
  popops::encodeOneHot(graph(), label1D, oneHot, prog, debugPrefix("nll"));

  // -1 at position "label", 0 elsewhere.
  // p - 1 at position "label" label, p elsewhere.
  popops::mapInPlace(graph(),
                     pe::Add(pe::Neg(pe::_1), pe::_2),
                     {oneHot, probs2D},
                     prog,
                     debugPrefix("negsub"));

  if (sfmgd.hasIgnoreIndex()) {
    auto lossMask = NllOpx::applyMaskInPlaceForIgnoredIndex(
        *this, graph(), oneHot, label1D, sfmgd.getIgnoreIndex(), prog);
    if (sfmgd.getReductionType() == ReductionType::Mean) {
      NllOpx::applyScalingInPlaceForMeanReductionWithIgnoreIndex(
          *this, graph(), oneHot, lossMask, prog);
    }
  } else {
    if (sfmgd.getReductionType() == ReductionType::Mean) {
      NllOpx::applyScalingInPlaceForMeanReduction(*this, graph(), oneHot, prog);
    }
  }

  // Output is reshaped to match probs input shape
  oneHot = oneHot.reshape(probs.shape());

  setOutTensor(0, oneHot);
}

// The maths for SoftmaxGradOp:
//   let L : any loss
//   p_i = sm (v_i) where sm is softmax
//   we define g_i = dL/dp_i
//   we want dL/dv_i
//   dL / dv_i = sum_j dL/dp_j dp_j/dv_i
//             = sum_j g_j [(i == j) p_j - p_i p_j]
//             = p_i g_i - p_i * sum_j ( p_j g_j)

void SoftmaxGradOpx::grow(poplar::program::Sequence &prog) const {
  const auto axis = getOp<SoftmaxGradOp>().getAxis();

  // The gradient of the loss w.r.t. the probabilities (g in above description)
  auto d_probs = getInTensor(SoftmaxGradOp::getGradProbsInIndex());
  d_probs      = coerceTo2D(d_probs, axis);

  // The input to the softmax (which we are computing the gradient of here)
  auto pre_probs = getInTensor(SoftmaxGradOp::getActsInIndex());
  pre_probs      = coerceTo2D(pre_probs, axis);

  // recomputing the probabilities (p in the above description)
  popnn::NonLinearityType nlType;
  if (dv_p->ir().getSessionOptions().enableNonStableSoftmax) {
    nlType = popnn::NonLinearityType::SOFTMAX;
  } else {
    nlType = popnn::NonLinearityType::SOFTMAX_STABLE;
  }
  auto probs = popnn::nonLinearity(
      graph(), nlType, pre_probs, prog, debugPrefix("nonLinearity"));

  // sum_j (p_j . g_j)
  // multiply probs by input gradient
  auto pg = popops::map(graph(),
                        popops::expr::BinaryOpType::MULTIPLY,
                        probs,
                        d_probs,
                        prog,
                        debugPrefix("mul"));

  // reduce along all dimensions except 0 (0 is the sample index)
  std::vector<size_t> redDims(probs.rank() - 1);
  std::iota(redDims.begin(), redDims.end(), 1);

  std::vector<size_t> upRanked(probs.rank(), 1);
  upRanked[0] = probs.dim(0);
  auto sum_pg = popops::reduce(graph(),
                               pg,
                               redDims,
                               {popops::Operation::ADD},
                               prog,
                               debugPrefix("reduce"))
                    .reshape(upRanked);

  auto dv = popops::map(graph(),
                        pe::Mul(pe::_1, pe::Sub(pe::_2, pe::_3)),
                        {probs, d_probs, sum_pg},
                        prog,
                        debugPrefix("SubMul"));

  dv = dv.reshape(inInfo(SoftmaxGradOp::getActsInIndex()).shape_szt());
  setOutTensor(0, dv);
}

void NlllWithSoftmaxGradDirectOpx::grow(poplar::program::Sequence &prog) const {
  NlllWithSoftmaxGradDirectOp &nllsfmgd = getOp<NlllWithSoftmaxGradDirectOp>();
  const poplar::Tensor &probs = getInTensor(NllLoss::getProbsInIndex());
  const poplar::Tensor &label = getInTensor(NllLoss::getLabelInIndex());

  // As for NllOpx, flatten outer dimensions if rank(probs) > 2
  auto probs2D = probs.flatten(0, probs.rank() - 1);
  auto label1D = label.flatten();

  // 1 at position "label", 0 elsewhere.
  // This tensor will be used for both NllLoss and SoftmaxDirectGrad
  auto oneHot =
      graph().clone(probs2D.elementType(), probs2D, debugPrefix("oneHot"));
  popops::encodeOneHot(graph(), label1D, oneHot, prog, debugPrefix("sfmGrad"));

  // First the first part of the loss calculation:

  // oneHotProbs, from a tensor which is sparse with a single 1 per row,
  //              to a tensor which is sparse with a single p per row.
  auto oneHotProbs = popops::map(graph(),
                                 popops::expr::BinaryOpType::MULTIPLY,
                                 oneHot,
                                 probs2D,
                                 prog,
                                 debugPrefix("mul"));

  // Now compute the SoftmaxGrad:

  // TODO: T8303
  // -1 at position "label", 0 elsewhere.
  // p - 1 at position "label" label, p elsewhere.
  popops::mapInPlace(graph(),
                     pe::Add(pe::Neg(pe::_1), pe::_2),
                     {oneHot, probs2D},
                     prog,
                     debugPrefix("NegSub"));

  if (nllsfmgd.hasIgnoreIndex()) {
    auto lossMask = NllOpx::applyMaskInPlaceForIgnoredIndex(
        *this, graph(), oneHot, label1D, nllsfmgd.getIgnoreIndex(), prog);
    if (nllsfmgd.getReductionType() == ReductionType::Mean) {
      NllOpx::applyScalingInPlaceForMeanReductionWithIgnoreIndex(
          *this, graph(), oneHot, lossMask, prog);
    }
  } else {
    if (nllsfmgd.getReductionType() == ReductionType::Mean) {
      NllOpx::applyScalingInPlaceForMeanReduction(*this, graph(), oneHot, prog);
    }
  }

  // Output is reshaped to match probs input shape
  oneHot = oneHot.reshape(probs.shape());

  // TODO T11263 base class to reduce code duplication
  // Apply loss scaling
  if (dv_p->ir().getOptimizer().lossScaling().isConst()) {
    auto lossScaling = dv_p->ir().getOptimizer().lossScaling().val();
    if (lossScaling > 1.0f || lossScaling < 1.0f) {
      popops::mapInPlace(graph(),
                         pe::Mul(pe::_1, pe::Const(lossScaling)),
                         {oneHot},
                         prog,
                         debugPrefix("scaledLoss"));
    }
  } else {
    popops::mapInPlace(
        graph(),
        pe::Mul(pe::_1, pe::_2),
        {oneHot,
         getInTensor(NlllWithSoftmaxGradDirectOp::getLossScalingInIndex())},
        prog,
        debugPrefix("scaledLoss"));
  }

  setOutTensor(nllsfmgd.getGradOutIndex(), oneHot);

  // Now compute the rest of the NllLoss from the same one-hot encoded tensor:

  // sum rows, so that just the p corresponding to the label remains
  poplar::Tensor reduction = popops::reduce(graph(),
                                            oneHotProbs,
                                            {1},
                                            {popops::Operation::ADD},
                                            prog,
                                            debugPrefix("add"));

  // Create an epsilon value
  poplar::Tensor eps =
      getConst(probs.elementType(), {1}, 1.0e-7, debugPrefix("epsilon"));
  // Add eps to reduction to make sure it does not have any 0's and log it,
  popops::mapInPlace(graph(),
                     pe::Log(pe::Add(pe::_1, pe::_2)),
                     {reduction, eps},
                     prog,
                     debugPrefix("LogEpsMul"));

  // TODO: T8305, re-use the mask created above
  if (nllsfmgd.hasIgnoreIndex()) {
    auto lossMask = NllOpx::applyMaskInPlaceForIgnoredIndex(
        *this, graph(), reduction, label1D, nllsfmgd.getIgnoreIndex(), prog);
    if (nllsfmgd.getReductionType() == ReductionType::Mean) {
      NllOpx::applyScalingInPlaceForMeanReductionWithIgnoreIndex(
          *this, graph(), reduction, lossMask, prog);
    }
  } else {
    if (nllsfmgd.getReductionType() == ReductionType::Mean) {
      NllOpx::applyScalingInPlaceForMeanReduction(
          *this, graph(), reduction, prog);
    }
  }

  // and negate it.
  popops::mapInPlace(graph(),
                     popops::expr::UnaryOpType::NEGATE,
                     reduction,
                     prog,
                     debugPrefix("neg"));

  // One loss per sample, so the output is reshaped to match label input shape
  reduction = reduction.reshape(label.shape());

  setOutTensor(nllsfmgd.getLossOutIndex(), reduction);
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
