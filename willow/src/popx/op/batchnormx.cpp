#include <poponnx/error.hpp>
#include <poponnx/ir.hpp>
#include <poponnx/op/batchnorm.hpp>
#include <poponnx/popx/devicex.hpp>
#include <poponnx/popx/op/batchnormx.hpp>
#include <poponnx/popx/opxmanager.hpp>

#include <poplar/Tensor.hpp>
#include <poplin/Norms.hpp>
#include <popnn/BatchNorm.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Expr.hpp>

namespace poplar {
using Shape = std::vector<std::size_t>;
}

namespace pe = popops::expr;

namespace poponnx {
namespace popx {

BatchNormOpx::BatchNormOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<BatchNormOp>(op, Onnx::Operators::BatchNormalization);
}

// convert variant to inverse standard deviation
static poplar::Tensor convertVarToInvSd(poplar::program::Sequence &prog,
                                        poplar::Graph &graph,
                                        const poplar::Tensor &var,
                                        float epsilon,
                                        std::string idStr) {
  // This will be replaced by the new operator Tim P is developing
  return popops::map(
      graph,
      pe::Divide(pe::Const(1), pe::Sqrt(pe::Add(pe::_1, pe::Const(epsilon)))),
      {var},
      prog,
      idStr + "/VarToInvSd");
}

// convert inverse standard deviation to variance
static poplar::Tensor convertInvSdToVar(poplar::program::Sequence &prog,
                                        poplar::Graph &graph,
                                        const poplar::Tensor &invSd,
                                        float epsilon,
                                        std::string idStr) {
  // This will be replaced by the new operator Tim P is developing
  return popops::map(
      graph,
      pe::Sub(pe::Divide(pe::Const(1), pe::Square(pe::_1)), pe::Const(epsilon)),
      {invSd},
      prog,
      idStr + "/InvSdToVar");
}

// Need to convert onnx input to poplar format. Poplar only excepts 2D or 4D
// tensors with the feature index in dimension 1.
// - For 4D tensors we are already in the right format
// - For nD tensors we can flatten to a 2D {X, C}
static std::pair<poplar::Tensor, poplar::Shape>
convertOnnxInputToPoplarInput(const poplar::Tensor &onnxInput) {
  poplar::Tensor poplarInput;
  poplar::Shape nonBroadcastDimensions;
  if (onnxInput.rank() == 4) {
    // The channels are already in dim 1, nothing to do
    poplarInput = onnxInput;
  } else {
    const unsigned finalDimension = onnxInput.rank() - 1;
    poplarInput            = onnxInput.dimShufflePartial({1}, {finalDimension});
    nonBroadcastDimensions = poplarInput.shape();
    nonBroadcastDimensions.pop_back();

    auto count  = onnxInput.numElements() / onnxInput.dim(1);
    poplarInput = poplarInput.reshapePartial(0, finalDimension, {count});
  }

  return {poplarInput, nonBroadcastDimensions};
}

// Convert back from poplar format to onnx format
static poplar::Tensor
convertPoplarOutputToOnnxOutput(const poplar::Tensor &poplarOutput,
                                const poplar::Shape &nonBroadcastDimensions) {
  poplar::Tensor onnxOutput;

  if (poplarOutput.rank() == 4) {
    // The channels are already in dim 1, nothing to do
    onnxOutput = poplarOutput;
  } else {
    onnxOutput = poplarOutput.reshapePartial(0, 1, {nonBroadcastDimensions});
    const unsigned finalDimension = onnxOutput.rank() - 1;
    onnxOutput = onnxOutput.dimShufflePartial({finalDimension}, {1});
  }

  return onnxOutput;
}

// Not clear to me if batchNormalise is meant update the mean/var then how can
// they be constant tensors
poplar::Tensor BatchNormOpx::batchNormalise(poplar::program::Sequence &prog,
                                            const poplar::Tensor &x,
                                            const poplar::Tensor &scale,
                                            const poplar::Tensor &b,
                                            const poplar::Tensor &mean,
                                            const poplar::Tensor &invSd) const {

  //  combinedMultiplicand = gamma / sDev
  //                       = gamma * invSd
  auto multiplcand = popops::map(graph(),
                                 pe::Mul(pe::_1, pe::_2),
                                 {scale, invSd},
                                 prog,
                                 idStr() + "/Multiplicand");

  // addend = beta - gamma * mean / sdDev
  //        = beta - gamma * mean * invSd
  auto addend = popops::map(graph(),
                            pe::Sub(pe::_1, pe::Mul(pe::_2, pe::_3)),
                            {b, multiplcand, mean},
                            prog,
                            idStr() + "/Addend");

  // Perform the batchNorm
  return popnn::bn::batchNormalise(
      graph(), x, multiplcand, addend, prog, idStr());
}

static bool isZeroElementArray(const poplar::Shape &shape) {
  return std::all_of(
      shape.begin(), shape.end(), [](int dim) -> bool { return dim == 0; });
}

void BatchNormOpx::grow(poplar::program::Sequence &prog) const {

  // OK. This is found more by trial and error. It appears that pytorch, is
  // using an unbiased running variance But the other question is what should we
  // do for onnx that does not state if the running_variance is biased or
  // unbiased. The ONNX calculation for running_xxx is different than pyTorch

  // Using the input/attribute names as per the onnx spec

  // Inputs
  auto x     = get(inId(BatchNormOp::getXInIndex()));
  auto scale = get(inId(BatchNormOp::getScaleInIndex()));
  auto b     = get(inId(BatchNormOp::getBInIndex()));
  auto mean  = get(inId(BatchNormOp::getMeanInIndex()));
  auto var   = get(inId(BatchNormOp::getVarInIndex()));

  // Attributes
  float epsilon  = getOp<BatchNormOp>().getEpsilon();
  float momentum = getOp<BatchNormOp>().getMomentum();
  // int spatial = getOp<BatchNormOp>().getSpatial();

  if (getOp<BatchNormOp>().isTraining()) {

    // Special case - zero sized array
    if (isZeroElementArray(x.shape())) {
      auto y         = graph().addConstant(x.elementType(), x.shape(), 0);
      auto batchMean = graph().addConstant(x.elementType(), {1}, NAN);
      auto batchVar  = graph().addConstant(x.elementType(), {1}, NAN);
      graph().setTileMapping(y, 0);
      graph().setTileMapping(batchMean, 0);
      graph().setTileMapping(batchVar, 0);
      insert(outId(BatchNormOp::getYOutIndex()), y);
      insert(outId(BatchNormOp::getMeanOutIndex()), batchMean);
      insert(outId(BatchNormOp::getVarOutIndex()), batchVar);
    } else {
      // Convert input shape to poplar rules
      poplar::Tensor xP;
      poplar::Shape nonBroadcastDims;
      std::tie(xP, nonBroadcastDims) = convertOnnxInputToPoplarInput(x);

      poplar::Tensor batchMean, invSd;
      std::tie(batchMean, invSd) = popnn::bn::batchNormStatistics(
          graph(), xP, epsilon, prog, false, poplar::FLOAT, idStr());

      // batch normalise
      auto y = batchNormalise(prog, xP, scale, b, batchMean, invSd);

      // convert updated inverse standard deviation back to variance.

      /*
      // Left this code in for now which computes the variance used in pytorch
      {
        // First we have to convert the invSd to the unbiased version
        const auto numElements = xP.numElements() / xP.dim(1);
        invSd = popops::map(graph(),
                                  pe::Mul(pe::_1,
      pe::Sqrt(pe::Divide(pe::Const(numElements -1),pe::Const(numElements)))),
                                  {invSd},
                                  prog,
                                  idStr() + "/unbiasedInvSd");
      }
      */

      // Then convert the invSd to the variance
      auto batchVar = convertInvSdToVar(prog, graph(), invSd, epsilon, idStr());

      // Convert the output back into the input format
      y = convertPoplarOutputToOnnxOutput(y, nonBroadcastDims);

      // Calculate the running mean
      auto runningMean = popops::map(
          graph(),
          pe::Add(pe::Mul(pe::Sub(pe::Const(1), pe::Const(momentum)), pe::_2),
                  pe::Mul(pe::Const(momentum), pe::_1)),
          {mean, batchMean},
          prog,
          idStr() + "/runningMean");

      // Calculate the running variance using the unbiased results
      auto runningVar = popops::map(
          graph(),
          pe::Add(pe::Mul(pe::Sub(pe::Const(1), pe::Const(momentum)), pe::_2),
                  pe::Mul(pe::Const(momentum), pe::_1)),
          {var, batchVar},
          prog,
          idStr() + "/runningVar");

      // return the results
      insert(outId(BatchNormOp::getYOutIndex()), y);
      insert(outId(BatchNormOp::getMeanOutIndex()), runningMean);
      insert(outId(BatchNormOp::getVarOutIndex()), runningVar);
      insert(outId(BatchNormOp::getSavedMeanOutIndex()), batchMean);
      insert(outId(BatchNormOp::getSavedVarOutIndex()), batchVar);
    }
  } else {
    // When testing

    // Special case - zero sized array
    if (isZeroElementArray(x.shape())) {
      auto y = graph().addConstant(x.elementType(), x.shape(), 0);
      graph().setTileMapping(y, 0);
      insert(outId(BatchNormOp::getYOutIndex()), y);
    } else {
      // Convert input shape to poplar rules
      poplar::Tensor xP;
      poplar::Shape nonBroadcastDims;
      std::tie(xP, nonBroadcastDims) = convertOnnxInputToPoplarInput(x);

      // convert variant to inverse standard deviation
      auto invSd = convertVarToInvSd(prog, graph(), var, epsilon, idStr());

      // batchnorm
      auto y = batchNormalise(prog, xP, scale, b, mean, invSd);

      // Convert the output back into the input format
      y = convertPoplarOutputToOnnxOutput(y, nonBroadcastDims);

      // return the result
      insert(outId(BatchNormOp::getYOutIndex()), y);
    }
  }
}

std::tuple<poplar::Tensor, poplar::Tensor, poplar::Tensor>
BatchNormGradOpx::batchNormaliseGrad(poplar::program::Sequence &prog,
                                     const poplar::Tensor &x,
                                     const poplar::Tensor &scale,
                                     const poplar::Tensor &mean,
                                     const poplar::Tensor &invSd,
                                     const poplar::Tensor &yGrad) const {

  poplar::Tensor xGrad, scaleGrad, bGrad;

  poplar::Tensor xWhitened = popnn::bn::batchNormWhiten(
      graph(), x, mean, invSd, prog, idStr() + "/WhitenedActs");

  // Compute the delta for the operand
  xGrad = popnn::bn::batchNormGradients(graph(),
                                        xWhitened,
                                        yGrad,
                                        invSd,
                                        scale,
                                        prog,
                                        poplar::FLOAT,
                                        idStr() + "/OperandGrad");

  // Compute the deltas for scaled and offset
  std::tie(scaleGrad, bGrad) =
      popnn::bn::batchNormParamGradients(graph(),
                                         xWhitened,
                                         yGrad,
                                         prog,
                                         poplar::FLOAT,
                                         idStr() + "/ScaleOffsetGrads");

  return std::make_tuple(xGrad, scaleGrad, bGrad);
}

BatchNormGradOpx::BatchNormGradOpx(Op *op, Devicex *devicex)
    : Opx(op, devicex) {
  verifyOp<BatchNormGradOp>(op, Onnx::GradOperators::BatchNormalizationGrad);
}

void BatchNormGradOpx::grow(poplar::program::Sequence &prog) const {
  // Inputs
  auto x     = get(inId(BatchNormGradOp::getXInIndex()));
  auto scale = get(inId(BatchNormGradOp::getScaleInIndex()));
  auto mean  = get(inId(BatchNormGradOp::getMeanInIndex()));
  auto var   = get(inId(BatchNormGradOp::getVarInIndex()));
  auto yGrad = get(inId(BatchNormGradOp::getYGradInIndex()));

  // Attributes
  float epsilon = getOp<BatchNormGradOp>().getFwdOp()->getEpsilon();
  // float momentum = getOp<BatchNormGradOp>().getFwdOp()->getMomentum();
  // int spatial = getOp<BatchNormGradOp>().getFwdOp()->getSpatial();

  // Special case - zero sized array
  if (isZeroElementArray(x.shape())) {
    auto xGrad     = graph().addConstant(x.elementType(), x.shape(), 0);
    auto scaleGrad = graph().addConstant(x.elementType(), {1}, 0);
    auto bGrad     = graph().addConstant(x.elementType(), {1}, 0);
    graph().setTileMapping(xGrad, 0);
    graph().setTileMapping(scaleGrad, 0);
    graph().setTileMapping(bGrad, 0);
    insert(outId(BatchNormGradOp::getXOutIndex()), xGrad);
    insert(outId(BatchNormGradOp::getScaleOutIndex()), scaleGrad);
    insert(outId(BatchNormGradOp::getBOutIndex()), bGrad);
  } else {

    // Convert input's shape to poplar rules
    poplar::Tensor xP, yGradP;
    poplar::Shape nonBroadcastDims;
    std::tie(xP, nonBroadcastDims)     = convertOnnxInputToPoplarInput(x);
    std::tie(yGradP, nonBroadcastDims) = convertOnnxInputToPoplarInput(yGrad);

    auto invSd = convertVarToInvSd(prog, graph(), var, epsilon, idStr());

    // batchnormgrad
    poplar::Tensor xGrad, scaleGrad, bGrad;
    std::tie(xGrad, scaleGrad, bGrad) =
        batchNormaliseGrad(prog, xP, scale, mean, invSd, yGradP);

    // Convert the output back into the input format
    xGrad = convertPoplarOutputToOnnxOutput(xGrad, nonBroadcastDims);

    // return the results
    insert(outId(BatchNormGradOp::getXOutIndex()), xGrad);
    insert(outId(BatchNormGradOp::getScaleOutIndex()), scaleGrad);
    insert(outId(BatchNormGradOp::getBOutIndex()), bGrad);
  }
}

namespace {
OpxCreator<BatchNormOpx>
    batchNormOpxCreator(Onnx::Operators::BatchNormalization);
OpxCreator<BatchNormGradOpx>
    batchNormGradOpxCreator(Onnx::GradOperators::BatchNormalizationGrad);
} // namespace

} // namespace popx
} // namespace poponnx
