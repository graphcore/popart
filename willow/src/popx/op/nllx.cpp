#include <memory>
#include <poponnx/error.hpp>
#include <poponnx/op/nll.hpp>
#include <poponnx/popx/devicex.hpp>
#include <poponnx/popx/op/nllx.hpp>
#include <poponnx/popx/opxmanager.hpp>

#include <popops/Cast.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Encoding.hpp>
#include <popops/Reduce.hpp>

namespace poponnx {
namespace popx {

NllOpx::NllOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<NllOp>(op, Onnx::CustomOperators::Nll);
}

void NllOpx::grow(poplar::program::Sequence &prog) const {
  const NllLoss *nllloss      = getOp<NllOp>().nlll();
  const poplar::Tensor &probs = get(nllloss->probsTensorId());
  const poplar::Tensor &label = get(nllloss->labelTensorId());

  // Expect an N-d Probs tensor and (N-1)-d Label tensor. If N=2:
  // Probs - a tensor of shape [Batchsize, NumClasses]
  // Label - a tensor of shape [Batchsize], where each element is a
  //         class index
  // If N > 2, then the inputs are flattened across all dimenions
  // (except the outer Classes dim in the case of Probs)
  auto probs2D = probs.flatten(0, probs.rank() - 1);
  auto label1D = label.flatten();

  // Tensor taking one-hot encoded output must be 2 dimensional
  auto oneHot =
      graph().clone(probs2D.elementType(), probs2D, debugPrefix("OneHot"));
  popops::encodeOneHot(graph(), label1D, oneHot, prog, debugPrefix("Nll"));

  // oneHot, from a tensor which is sparse with a single 1 per row,
  //           to a tensor which is sparse with a single p per row.
  popops::mapInPlace(graph(),
                     popops::expr::BinaryOpType::MULTIPLY,
                     oneHot,
                     probs2D,
                     prog,
                     debugPrefix("Mul"));

  // sum rows, so that just the p corresponding to the label remains
  poplar::Tensor reduction =
      popops::reduce(graph(), oneHot, {1}, {popops::Operation::ADD}, prog);

  // Create an epsilon value
  poplar::Tensor eps =
      getConst(probs.elementType(), {1}, 1.0e-7, debugPrefix("epsilon"));

  // Add eps to reduction make sure it does not have any 0's before the
  // log
  reduction = popops::map(graph(),
                          popops::expr::BinaryOpType::ADD,
                          reduction,
                          eps,
                          prog,
                          debugPrefix("EpsMul"));
  // and log it,
  popops::mapInPlace(graph(),
                     popops::expr::UnaryOpType::LOGARITHM,
                     reduction,
                     prog,
                     debugPrefix("Log"));

  if (nllloss->hasIgnoreIndex()) {
    auto lossMask = applyMaskInPlaceForIgnoredIndex(
        *this, graph(), reduction, label1D, nllloss->getIgnoreIndex(), prog);
    if (nllloss->getReductionType() == ReductionType::MEAN) {
      applyScalingInPlaceForMeanReductionWithIgnoreIndex(
          *this, graph(), reduction, lossMask, prog);
    }
  } else {
    if (nllloss->getReductionType() == ReductionType::MEAN) {
      applyScalingInPlaceForMeanReduction(*this, graph(), reduction, prog);
    }
  }

  // and negate it.
  popops::mapInPlace(graph(),
                     popops::expr::UnaryOpType::NEGATE,
                     reduction,
                     prog,
                     debugPrefix("Neg"));

  // One loss per sample, so the output is reshaped to match label input shape
  reduction = reduction.reshape(label.shape());

  setOutTensor(0, reduction);
}

void NllOpx::applyScalingInPlaceForMeanReduction(
    const Opx &opx,
    poplar::Graph &graph,
    poplar::Tensor t,
    poplar::program::Sequence &prog) {
  double totalSamples = static_cast<double>(opx.dv_p->getReplicationFactor()) *
                        static_cast<double>(t.dim(0));
  auto t_totalSamples = opx.getConst(
      t.elementType(), {}, totalSamples, opx.debugPrefix("Samples"));
  popops::mapInPlace(graph,
                     popops::expr::BinaryOpType::DIVIDE,
                     t,
                     t_totalSamples,
                     prog,
                     opx.debugPrefix("Mean"));
}

void NllOpx::applyScalingInPlaceForMeanReductionWithIgnoreIndex(
    const Opx &opx,
    poplar::Graph &graph,
    poplar::Tensor t,
    poplar::Tensor mask,
    poplar::program::Sequence &prog) {
  // Determine the scale-factor for mean reduction dynamically from the
  // mask.
  // Any sample whose label index is the 'ignore index' should not be
  // counted when scaling the loss/loss grad
  if (mask.rank() == 2 && mask.dim(1) == 1) {
    mask = mask.squeeze({1});
  }
  auto numNonIgnoredSamples =
      popops::reduce(graph, mask, {0}, {popops::Operation::ADD}, prog);

  auto repFactor =
      opx.getConst(t.elementType(),
                   {},
                   static_cast<double>(opx.dv_p->getReplicationFactor()),
                   opx.debugPrefix("ReplicationFactor"));

  auto totalSamples = popops::map(graph,
                                  popops::expr::BinaryOpType::MULTIPLY,
                                  repFactor,
                                  numNonIgnoredSamples,
                                  prog,
                                  opx.debugPrefix("TotalSamples"));

  popops::mapInPlace(graph,
                     popops::expr::BinaryOpType::DIVIDE,
                     t,
                     totalSamples,
                     prog,
                     opx.debugPrefix("Mean"));
}

poplar::Tensor
NllOpx::applyMaskInPlaceForIgnoredIndex(const Opx &opx,
                                        poplar::Graph &graph,
                                        poplar::Tensor t,
                                        poplar::Tensor labels,
                                        int ignoreIndex,
                                        poplar::program::Sequence &prog) {
  // Get the scalar ignoreIndex tensor. If it doens't already
  // exist, create it
  auto ignoreIndexTensor = graph.addConstant(
      labels.elementType(), {}, ignoreIndex, opx.debugPrefix("ignoreIndex"));
  graph.setTileMapping(ignoreIndexTensor, 0);

  // Create the mask
  auto lossMaskBool = popops::map(graph,
                                  popops::expr::BinaryOpType::NOT_EQUAL,
                                  labels,
                                  ignoreIndexTensor,
                                  prog,
                                  opx.debugPrefix("NotEqual"));
  auto lossMask     = popops::cast(
      graph, lossMaskBool, t.elementType(), prog, opx.debugPrefix("Cast"));

  // Expand, if required, for valid broadcasting of mul
  if (t.rank() == 2) {
    lossMask = lossMask.expand({1});
  }

  // Apply the mask
  popops::mapInPlace(graph,
                     popops::expr::BinaryOpType::MULTIPLY,
                     t,
                     lossMask,
                     prog,
                     opx.debugPrefix("masked"));

  return lossMask;
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
  const NllLoss *nllloss      = getOp<NllGradOp>().nlll();
  const poplar::Tensor &probs = get(nllloss->probsTensorId());
  const poplar::Tensor &label = get(nllloss->labelTensorId());

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
                               debugPrefix("Max"));

  // oneHot: initialised to be 1 at position "label", 0 elsewhere.
  auto oneHot =
      graph().clone(probs2D.elementType(), probs2D, debugPrefix("OneHot"));

  popops::encodeOneHot(graph(), label1D, oneHot, prog, debugPrefix("Nll"));

  // oneHot: becomes -1 at position "label", 0 elsewhere.
  popops::mapInPlace(graph(),
                     popops::expr::UnaryOpType::NEGATE,
                     oneHot,
                     prog,
                     debugPrefix("Neg"));

  // oneHot: set to -1/p at position "label", 0 elsewhere.
  popops::mapInPlace(graph(),
                     popops::expr::BinaryOpType::DIVIDE,
                     oneHot,
                     safeProbs,
                     prog,
                     debugPrefix("Div"));

  // Apply mask before reduction, so that ignored class doesn't
  // contribute to the loss gradient
  if (nllloss->hasIgnoreIndex()) {
    auto lossMask = NllOpx::applyMaskInPlaceForIgnoredIndex(
        *this, graph(), oneHot, label1D, nllloss->getIgnoreIndex(), prog);
    if (nllloss->getReductionType() == ReductionType::MEAN) {
      NllOpx::applyScalingInPlaceForMeanReductionWithIgnoreIndex(
          *this, graph(), oneHot, lossMask, prog);
    }
  } else {
    if (nllloss->getReductionType() == ReductionType::MEAN) {
      NllOpx::applyScalingInPlaceForMeanReduction(*this, graph(), oneHot, prog);
    }
  }

  // Output is reshaped to match probs input shape
  oneHot = oneHot.reshape(probs.shape());

  setOutTensor(0, oneHot);
}

namespace {
static OpxCreator<NllOpx> nllOpxCreator(Onnx::CustomOperators::Nll);
static OpxCreator<NllGradOpx>
    nllGradOpxCreator(Onnx::CustomGradOperators::NllGrad);
} // namespace

} // namespace popx
} // namespace poponnx
