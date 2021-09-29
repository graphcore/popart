// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <popart/error.hpp>
#include <popart/op/logsoftmax.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/logsoftmaxx.hpp>
#include <popart/popx/opxmanager.hpp>

#include <popnn/LogSoftmax.hpp>
#include <popnn/NonLinearity.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Expr.hpp>
#include <popops/Reduce.hpp>

namespace pe = popops::expr;

namespace popart {
namespace popx {

template <typename T>
std::unique_ptr<LogSoftmaxComputex> createLogSoftmaxComputex(Op *op) {
  auto lsmop = dynamic_cast<T *>(op);
  if (lsmop == nullptr) {
    throw error("Cannot create LogSoftmaxComputex from {}", op->str());
  }

  int64_t axis         = lsmop->getAxis();
  const auto &outShape = lsmop->outInfo(lsmop->getOutIndex()).shape_szt();
  return std::make_unique<LogSoftmaxComputex>(axis, outShape);
}

LogSoftmaxOpx::LogSoftmaxOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryOutplaceOpx(op,
                                  devicex,
                                  createLogSoftmaxComputex<LogSoftmaxOp>(op)) {}

LogSoftmaxInplaceOpx::LogSoftmaxInplaceOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryInplaceOpx(
          op,
          devicex,
          createLogSoftmaxComputex<LogSoftmaxInplaceOp>(op)) {}

snap::Tensor LogSoftmaxComputex::outplace(snap::program::Sequence &p,
                                          snap::Graph &g,
                                          const snap::Tensor &t,
                                          const poplar::DebugNameAndId &dnai,
                                          const std::string &s) const {
  auto outTensor = cloneNcopy(p, g, t, dnai);
  inplace(p, g, outTensor, dnai, s);
  return outTensor;
}

void LogSoftmaxComputex::inplace(snap::program::Sequence &p,
                                 snap::Graph &g,
                                 const snap::Tensor &t,
                                 const poplar::DebugNameAndId &dnai,
                                 const std::string &s) const {
  popnn::logSoftmaxInPlace(g.getPoplarGraph(),
                           coerceTo2D(t, axis).getPoplarTensor(),
                           p.getPoplarSequence(),
                           {dnai, s});
}

snap::Tensor LogSoftmaxComputex::reshape(const snap::Tensor &t) const {
  return t.reshape(outShape);
}

LogSoftmaxGradOpx::LogSoftmaxGradOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryOpx(op, devicex) {
  verifyOp<LogSoftmaxGradOp>(op, Onnx::GradOperators::LogSoftmaxGrad);
}

// Note that we cannot just use 1/softmax(x_i) * softmaxgrad(x_i) since
// softmax(x_i) might be zero.
//
// Let softmax(x_i) = exp(x_i) / Z where Z = sum_i exp(x_i)
//
// we can show that logsoftmax(x_i) = x_i - log(Z)
//
// d/dx_i logsoftmax(x_j) = (i==j) 1 - softmax(x_j)
//                          (i!=j)    -softmax(x_j)
//
// Given L as any loss and y_i = logsoftmax(x_i)
//
// dL/dy_i := g_i (gradient of the loss w.r.t the logsoftmax)
//
// we want dL/dx_i = sum_j dL/dy_j * dy_j/dx_i
//                 = g_i - softmax(x_i) * sum_j g_j
void LogSoftmaxGradOpx::grow(snap::program::Sequence &prog) const {
  const auto axis = getOp<LogSoftmaxGradOp>().getAxis();

  // The gradient of the loss w.r.t. the probabilities (g in above description)
  auto d_probs = getInTensor(LogSoftmaxGradOp::getGradProbsInIndex());
  d_probs      = EwuComputex::coerceTo2D(d_probs, axis);

  // The input to the logsoftmax (which we are computing the gradient of here)
  auto pre_probs = getInTensor(LogSoftmaxGradOp::getActsInIndex());
  pre_probs      = EwuComputex::coerceTo2D(pre_probs, axis);

  // compute the probabilities from softmax
  popnn::NonLinearityType nlType;
  if (dv_p->ir().getSessionOptions().enableNonStableSoftmax) {
    nlType = popnn::NonLinearityType::SOFTMAX;
  } else {
    nlType = popnn::NonLinearityType::SOFTMAX_STABLE;
  }
  auto probs = popnn::nonLinearity(graph().getPoplarGraph(),
                                   nlType,
                                   pre_probs.getPoplarTensor(),
                                   prog.getPoplarSequence(),
                                   debugContext("nonLinearity"));

  // sum_j (g_j)
  // reduce along all dimensions except 0 (0 is the sample index)
  std::vector<size_t> redDims(probs.rank() - 1);
  std::iota(redDims.begin(), redDims.end(), 1);

  std::vector<size_t> upRanked(probs.rank(), 1);
  upRanked[0] = probs.dim(0);
  auto sum_g  = popops::reduce(graph().getPoplarGraph(),
                              d_probs.getPoplarTensor(),
                              redDims,
                              {popops::Operation::ADD},
                              prog.getPoplarSequence(),
                              debugContext("reduce"))
                   .reshape(upRanked);

  // g_i - softmax(x_i) * sum_j (g_j)
  auto dv = popops::map(graph().getPoplarGraph(),
                        pe::Sub(pe::_1, pe::Mul(pe::_2, pe::_3)),
                        {d_probs.getPoplarTensor(), probs, sum_g},
                        prog.getPoplarSequence(),
                        debugContext("SubMul"));

  dv = dv.reshape(inInfo(LogSoftmaxGradOp::getActsInIndex()).shape_szt());
  setOutTensor(0, snap::Tensor{dv, graph()});
}

namespace {
OpxCreator<LogSoftmaxOpx> logSoftmaxOpxCreator(
    {Onnx::Operators::LogSoftmax_1, Onnx::Operators::LogSoftmax_11});

OpxCreator<LogSoftmaxInplaceOpx>
    logSoftmaxInplaceOpxCreator({Onnx::CustomOperators::LogSoftmaxInplace});

OpxCreator<LogSoftmaxGradOpx>
    logSoftmaxGradOpxCreator({Onnx::GradOperators::LogSoftmaxGrad});
} // namespace

} // namespace popx
} // namespace popart
