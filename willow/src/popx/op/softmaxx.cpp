#include <poponnx/error.hpp>
#include <poponnx/makeunique.hpp>
#include <poponnx/op/nll.hpp>
#include <poponnx/op/softmax.hpp>
#include <poponnx/popx/op/softmaxx.hpp>
#include <poponnx/popx/opxmanager.hpp>

#include "popops/Encoding.hpp"
#include <popnn/NonLinearity.hpp>
#include <popops/ElementWise.hpp>

namespace poponnx {
namespace popx {

SoftmaxOpx::SoftmaxOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<SoftmaxOp>(op, Onnx::Operators::Softmax);
}

void SoftmaxOpx::grow(poplar::program::Sequence &prog) const {

  auto outTensor = popnn::nonLinearity(
      graph(), popnn::NonLinearityType::SOFTMAX, get(inId(0)), prog, outId(0));

  insert(outId(0), outTensor);
}

SoftmaxGradOpx::SoftmaxGradOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<SoftmaxGradOp>(op, Onnx::GradOperators::SoftmaxGrad);
}

SoftmaxGradDirectOpx::SoftmaxGradDirectOpx(Op *op, Devicex *devicex)
    : Opx(op, devicex) {
  verifyOp<SoftmaxGradDirectOp>(op,
                                Onnx::CustomGradOperators::SoftmaxGradDirect);
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

void SoftmaxGradDirectOpx::grow(poplar::program::Sequence &prog) const {
  SoftmaxGradDirectOp &sfmgd = getOp<SoftmaxGradDirectOp>();
  TensorId labelId           = sfmgd.nlll()->labelTensorId();
  TensorId probsId           = sfmgd.nlll()->probsTensorId();

  // 1 at position "label", 0 elsewhere.
  auto oneHot =
      graph().clone(get(probsId).elementType(), get(probsId), "..OneHot");
  popops::encodeOneHot(graph(), get(labelId), oneHot, prog, "..Nll");
  // -1 at position "label", 0 elsewhere.
  popops::mapInPlace(
      graph(), popops::expr::UnaryOpType::NEGATE, oneHot, prog, "..neg");

  // p - 1 at position "label" label, p elsewhere.
  popops::mapInPlace(graph(),
                     popops::expr::BinaryOpType::ADD,
                     oneHot,
                     get(probsId),
                     prog,
                     "..sub");

  insert(outId(0), oneHot);
}

namespace {
OpxCreator<SoftmaxOpx> softmaxOpxCreator(Onnx::Operators::Softmax);
OpxCreator<SoftmaxGradOpx>
    softmaxGradOpxCreator(Onnx::GradOperators::SoftmaxGrad);
OpxCreator<SoftmaxGradDirectOpx>
    softmaxGradDirectOpxCreator(Onnx::CustomGradOperators::SoftmaxGradDirect);
} // namespace

} // namespace popx
} // namespace poponnx
