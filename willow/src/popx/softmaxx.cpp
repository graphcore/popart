#include <poponnx/error.hpp>
#include <poponnx/nll.hpp>
#include <poponnx/popx/softmaxx.hpp>
#include <poponnx/softmax.hpp>
#include <poponnx/util.hpp>

#pragma clang diagnostic push // start ignoring warnings
#pragma clang diagnostic ignored "-Weverything"
#include "popops/Encoding.hpp"
#include <popnn/NonLinearity.hpp>
#include <popops/ElementWise.hpp>
#pragma clang diagnostic pop // stop ignoring warnings

namespace willow {
namespace popx {

SoftmaxOpx::SoftmaxOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  if (op->opType != OpType::SOFTMAX) {
    throw error("cannot create SoftmaxOpx from " + op->op_type());
  }
}

void SoftmaxOpx::grow() const {

  auto outTensor = popnn::nonLinearity(graph(),
                                       popnn::NonLinearityType::SOFTMAX,
                                       get(inId(0)),
                                       step(),
                                       outId(0));

  insert(outId(0), outTensor);
}

SoftmaxOp *SoftmaxOpx::getSoftmaxOp() const {
  return dynamic_cast<SoftmaxOp *>(op_p);
}

SoftmaxGradOpx::SoftmaxGradOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  if (op->opType != OpType::SOFTMAXGRAD) {
    throw error("cannot create SoftmaxGradOpx from " + op->op_type());
  }
}

SoftmaxGradOp *SoftmaxGradOpx::getSoftmaxGradOp() const {
  return dynamic_cast<SoftmaxGradOp *>(op_p);
}

SoftmaxGradDirectOpx::SoftmaxGradDirectOpx(Op *op, Devicex *devicex)
    : Opx(op, devicex) {
  if (op->opType != OpType::SOFTMAXGRADDIRECT) {
    throw error("cannot create SoftmaxGradDirectOpx from " + op->op_type());
  }
}

SoftmaxGradDirectOp *SoftmaxGradDirectOpx::getSoftmaxGradDirectOp() const {
  return dynamic_cast<SoftmaxGradDirectOp *>(op_p);
}

// The maths:
// loss = -ln(p_j), where j is the true class
// d(loss)/d(p_i) = 0, d(loss)/d(p_j) = -1/p_j
// p_j = exp(v_j) / S
// where S = sum_{all indices k} [ exp(v_k) ]
// By the quotient rule:
// d(p_j)/d(v_i)  = (0 - exp(v_j).exp(v_i)) / S^2
//                = -p_i.p_j
// d(p_j)/d(v_j)  = (exp(v_j).S - exp(v_j).exp(v_j)) / S^2
//                = p_j - p_i.p_j
// Then, using the chain rule,
// d(loss)/d(v_i) = p_i
// d(loss)/d(v_j) = p_j - 1
//
// -----
// |   |
// |   |
// -----

void SoftmaxGradDirectOpx::grow() const {
  SoftmaxGradDirectOp *sfmgd = getSoftmaxGradDirectOp();
  TensorId labelId           = sfmgd->nlll()->labelTensorId();
  TensorId probsId           = sfmgd->nlll()->probsTensorId();

  // 1 at position "label", 0 elsewhere.
  auto oneHot =
      graph().clone(get(probsId).elementType(), get(probsId), "..OneHot");
  popops::encodeOneHot(graph(), get(labelId), oneHot, step(), "..Nll");
  // -1 at position "label", 0 elsewhere.
  popops::mapInPlace(
      graph(), popops::expr::UnaryOpType::NEGATE, oneHot, step(), "..neg");

  // p - 1 at position "label" label, p elsewhere.
  popops::mapInPlace(graph(),
                     popops::expr::BinaryOpType::ADD,
                     oneHot,
                     get(probsId),
                     step(),
                     "..sub");

  insert(outId(0), oneHot);
}

} // namespace popx
} // namespace willow
