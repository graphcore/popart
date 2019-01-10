#include <poponnx/error.hpp>
#include <poponnx/makeunique.hpp>
#include <poponnx/op/nll.hpp>
#include <poponnx/op/softmax.hpp>
#include <poponnx/popx/op/softmaxx.hpp>
#include <poponnx/popx/opxmanager.hpp>

#include <popnn/NonLinearity.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Encoding.hpp>
#include <popops/Reduce.hpp>

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

// The maths for SoftmaxGradOp:
//   let L : any loss
//   p_i = sm (v_i) where sm is softmax
//   we define g_i = dL/dp_i
//   we want dL/dv_i
//   dL / dv_i = sum_j dL/dp_j dp_j/dv_i
//             = sum_j g_j [(i == j) p_j - p_i p_j]
//             = p_i g_i - p_i * sum_j ( p_j g_j)

void SoftmaxGradOpx::grow(poplar::program::Sequence &prog) const {

  // The gradient of the loss w.r.t. the probabilities (g in above description)
  const auto &d_probs = get(inId(SoftmaxGradOp::getGradProbsInIndex()));

  // The input to the softmax (which we are computing the gradient of here)
  const auto &pre_probs = get(inId(SoftmaxGradOp::getActsInIndex()));

  // recomputing the probabilities (p in the above description)
  auto probs = popnn::nonLinearity(graph(),
                                   popnn::NonLinearityType::SOFTMAX,
                                   pre_probs,
                                   prog,
                                   "..SoftmaxGrad(0)");

  // sum_j (p_j . g_j)
  // multiply probs by input gradient
  auto pg = popops::map(graph(),
                        popops::expr::BinaryOpType::MULTIPLY,
                        probs,
                        d_probs,
                        prog,
                        "..SoftmaxGrad(1)");

  // reduce along all dimensions except 0 (0 is the sample index)
  std::vector<size_t> redDims(probs.rank() - 1);
  std::iota(redDims.begin(), redDims.end(), 1);

  std::vector<size_t> upRanked(probs.rank(), 1);
  upRanked[0] = probs.dim(0);
  auto sum_pg =
      popops::reduce(graph(), pg, redDims, {popops::Operation::ADD}, prog)
          .reshape(upRanked);

  auto g_minus_sum_pg = popops::map(graph(),
                                    popops::expr::BinaryOpType::SUBTRACT,
                                    d_probs,
                                    sum_pg,
                                    prog,
                                    "..SoftmaxGrad(2)");

  auto dv = popops::map(graph(),
                        popops::expr::BinaryOpType::MULTIPLY,
                        probs,
                        g_minus_sum_pg,
                        prog,
                        "..SoftmaxGrad(3)");

  insert(outId(0), dv);
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
