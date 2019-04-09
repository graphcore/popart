#include <iterator>
#include <poponnx/error.hpp>
#include <poponnx/ir.hpp>
#include <poponnx/makeunique.hpp>
#include <poponnx/op/nll.hpp>
#include <poponnx/op/softmax.hpp>
#include <poponnx/popx/devicex.hpp>
#include <poponnx/popx/op/nllx.hpp>
#include <poponnx/popx/op/softmaxx.hpp>
#include <poponnx/popx/opxmanager.hpp>

#include <popnn/NonLinearity.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Encoding.hpp>
#include <popops/Reduce.hpp>

namespace poponnx {
namespace popx {

SoftmaxOpx::SoftmaxOpx(Op *op, Devicex *devicex)
    : ElementWiseUnaryOpx(op, devicex) {
  verifyOp<SoftmaxOp>(op, Onnx::Operators::Softmax_1);
}

poplar::Tensor SoftmaxOpx::coerceTo2D(const poplar::Tensor &t, int64_t axis) {
  const auto in_shape = t.shape();
  auto k              = in_shape.begin();
  std::advance(k, axis);

  auto n = std::accumulate(
      in_shape.begin(), k, std::size_t{1}, std::multiplies<std::size_t>());
  auto d = std::accumulate(
      k, in_shape.end(), std::size_t{1}, std::multiplies<std::size_t>());
  return t.reshape({n, d});
}

void SoftmaxOpx::grow(poplar::program::Sequence &prog) const {
  auto input = getInTensor(SoftmaxOp::getInIndex());

  const auto axis = getOp<SoftmaxOp>().getAxis();
  input           = coerceTo2D(input, axis);

  // By default use stable softmax (prevent overflow by subtracting max
  // input value from input tensor before computing the exponentials).
  // Optionally override.
  popnn::NonLinearityType nlType;
  if (dv_p->ir().getSessionOptions().enableNonStableSoftmax) {
    nlType = popnn::NonLinearityType::SOFTMAX;
  } else {
    nlType = popnn::NonLinearityType::SOFTMAX_STABLE;
  }

  auto outTensor = popnn::nonLinearity(
      graph(), nlType, input, prog, debugPrefix("nonLinearity"));

  outTensor = outTensor.reshape(inInfo(SoftmaxOp::getInIndex()).shape_szt());
  setOutTensor(0, outTensor);
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
  const poplar::Tensor &probs = get(sfmgd.nlll()->probsTensorId());
  const poplar::Tensor &label = get(sfmgd.nlll()->labelTensorId());

  // As for NllOpx, flatten outer dimenstions if rank(probs) > 2
  auto probs2D = probs.flatten(0, probs.rank() - 1);
  auto label1D = label.flatten();

  // 1 at position "label", 0 elsewhere.
  auto oneHot =
      graph().clone(probs2D.elementType(), probs2D, debugPrefix("OneHot"));
  popops::encodeOneHot(graph(), label1D, oneHot, prog, debugPrefix("Nll"));

  // -1 at position "label", 0 elsewhere.
  popops::mapInPlace(graph(),
                     popops::expr::UnaryOpType::NEGATE,
                     oneHot,
                     prog,
                     debugPrefix("Neg"));

  // p - 1 at position "label" label, p elsewhere.
  popops::mapInPlace(graph(),
                     popops::expr::BinaryOpType::ADD,
                     oneHot,
                     probs2D,
                     prog,
                     debugPrefix("Sub"));

  if (sfmgd.nlll()->hasIgnoreIndex()) {
    NllOpx::applyMaskInPlaceForIgnoredIndex(
        *this, graph(), oneHot, label1D, sfmgd.nlll()->getIgnoreIndex(), prog);
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
  d_probs      = SoftmaxOpx::coerceTo2D(d_probs, axis);

  // The input to the softmax (which we are computing the gradient of here)
  auto pre_probs = getInTensor(SoftmaxGradOp::getActsInIndex());
  pre_probs      = SoftmaxOpx::coerceTo2D(pre_probs, axis);

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
                        debugPrefix("Mul"));

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
                               debugPrefix("Reduce"))
                    .reshape(upRanked);

  auto g_minus_sum_pg = popops::map(graph(),
                                    popops::expr::BinaryOpType::SUBTRACT,
                                    d_probs,
                                    sum_pg,
                                    prog,
                                    debugPrefix("Sub"));

  auto dv = popops::map(graph(),
                        popops::expr::BinaryOpType::MULTIPLY,
                        probs,
                        g_minus_sum_pg,
                        prog,
                        debugPrefix("Mul"));

  dv = dv.reshape(inInfo(SoftmaxGradOp::getActsInIndex()).shape_szt());
  setOutTensor(0, dv);
}

namespace {
OpxCreator<SoftmaxOpx> softmaxOpxCreator(Onnx::Operators::Softmax_1);
OpxCreator<SoftmaxGradOpx>
    softmaxGradOpxCreator(Onnx::GradOperators::SoftmaxGrad);
OpxCreator<SoftmaxGradDirectOpx>
    softmaxGradDirectOpxCreator(Onnx::CustomGradOperators::SoftmaxGradDirect);
} // namespace

} // namespace popx
} // namespace poponnx
