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

BatchNormOpx::BatchNormOpx(Op *op, Devicex *devicex) : NormOpx(op, devicex) {
  verifyOp<BatchNormOp>(op,
                        {Onnx::Operators::BatchNormalization_6,
                         Onnx::Operators::BatchNormalization_7,
                         Onnx::Operators::BatchNormalization_9});
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

  auto op = getOp<BatchNormOp>();

  // OK. This is found more by trial and error. It appears that pytorch, is
  // using an unbiased running variance But the other question is what should we
  // do for onnx that does not state if the running_variance is biased or
  // unbiased. The ONNX calculation for running_xxx is different than pyTorch

  // Using the input/attribute names as per the onnx spec

  // Inputs
  auto x     = getInTensor(BatchNormOp::getXInIndex());
  auto scale = getInTensor(BatchNormOp::getScaleInIndex());
  auto b     = getInTensor(BatchNormOp::getBInIndex());
  auto mean  = getInTensor(BatchNormOp::getMeanInIndex());
  auto var   = getInTensor(BatchNormOp::getVarInIndex());

  // Attributes
  float epsilon  = op.getEpsilon();
  float momentum = op.getMomentum();

  if (op.isTraining()) {

    // Special case - zero sized array
    if (isZeroElementArray(x.shape())) {
      auto y =
          graph().addConstant(x.elementType(), x.shape(), 0, debugPrefix("y"));
      auto batchMean =
          graph().addConstant(x.elementType(), {1}, NAN, debugPrefix("mean"));
      auto batchVar =
          graph().addConstant(x.elementType(), {1}, NAN, debugPrefix("var"));
      graph().setTileMapping(y, 0);
      graph().setTileMapping(batchMean, 0);
      graph().setTileMapping(batchVar, 0);
      setOutTensor(BatchNormOp::getYOutIndex(), y);
      setOutTensor(BatchNormOp::getMeanOutIndex(), batchMean);
      setOutTensor(BatchNormOp::getVarOutIndex(), batchVar);
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
      auto batchVar = convertInvSdToVar(prog, invSd, epsilon);

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
      setOutTensor(BatchNormOp::getYOutIndex(), y);
      setOutTensor(BatchNormOp::getMeanOutIndex(), runningMean);
      setOutTensor(BatchNormOp::getVarOutIndex(), runningVar);
      setOutTensor(BatchNormOp::getSavedMeanOutIndex(), batchMean);
      setOutTensor(BatchNormOp::getSavedVarOutIndex(), batchVar);
    }
  } else {
    // When testing

    // Special case - zero sized array
    if (isZeroElementArray(x.shape())) {
      auto y =
          graph().addConstant(x.elementType(), x.shape(), 0, debugPrefix("y"));
      graph().setTileMapping(y, 0);
      setOutTensor(BatchNormOp::getYOutIndex(), y);
    } else {
      // Convert input shape to poplar rules
      poplar::Tensor xP;
      poplar::Shape nonBroadcastDims;
      std::tie(xP, nonBroadcastDims) = convertOnnxInputToPoplarInput(x);

      // convert variant to inverse standard deviation
      auto invSd = convertVarToInvSd(prog, var, epsilon);

      // batchnorm
      auto y = batchNormalise(prog, xP, scale, b, mean, invSd);

      // Convert the output back into the input format
      y = convertPoplarOutputToOnnxOutput(y, nonBroadcastDims);

      // return the result
      setOutTensor(BatchNormOp::getYOutIndex(), y);
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
    : NormOpx(op, devicex) {
  verifyOp<BatchNormGradOp>(op, Onnx::GradOperators::BatchNormalizationGrad);
}

void BatchNormGradOpx::grow(poplar::program::Sequence &prog) const {

  auto op = getOp<BatchNormGradOp>();

  // Inputs
  auto x     = getInTensor(BatchNormGradOp::getXInIndex());
  auto scale = getInTensor(BatchNormGradOp::getScaleInIndex());
  auto mean  = getInTensor(BatchNormGradOp::getMeanInIndex());
  auto var   = getInTensor(BatchNormGradOp::getVarInIndex());
  auto yGrad = getInTensor(BatchNormGradOp::getYGradInIndex());

  // Attributes
  float epsilon = op.getEpsilon();

  // Special case - zero sized array
  if (isZeroElementArray(x.shape())) {
    auto xGrad = graph().addConstant(
        x.elementType(), x.shape(), 0, debugPrefix("xGrad"));
    auto scaleGrad =
        graph().addConstant(x.elementType(), {1}, 0, debugPrefix("scaleGrad"));
    auto bGrad =
        graph().addConstant(x.elementType(), {1}, 0, debugPrefix("bGrad"));
    graph().setTileMapping(xGrad, 0);
    graph().setTileMapping(scaleGrad, 0);
    graph().setTileMapping(bGrad, 0);
    setOutTensor(BatchNormGradOp::getXOutIndex(), xGrad);
    setOutTensor(BatchNormGradOp::getScaleOutIndex(), scaleGrad);
    setOutTensor(BatchNormGradOp::getBOutIndex(), bGrad);
  } else {

    // Convert input's shape to poplar rules
    poplar::Tensor xP, yGradP;
    poplar::Shape nonBroadcastDims;
    std::tie(xP, nonBroadcastDims)     = convertOnnxInputToPoplarInput(x);
    std::tie(yGradP, nonBroadcastDims) = convertOnnxInputToPoplarInput(yGrad);

    auto invSd = convertVarToInvSd(prog, var, epsilon);

    // batchnormgrad
    poplar::Tensor xGrad, scaleGrad, bGrad;
    std::tie(xGrad, scaleGrad, bGrad) =
        batchNormaliseGrad(prog, xP, scale, mean, invSd, yGradP);

    // Convert the output back into the input format
    xGrad = convertPoplarOutputToOnnxOutput(xGrad, nonBroadcastDims);

    // return the results
    setOutTensor(BatchNormGradOp::getXOutIndex(), xGrad);
    setOutTensor(BatchNormGradOp::getScaleOutIndex(), scaleGrad);
    setOutTensor(BatchNormGradOp::getBOutIndex(), bGrad);
  }
}

namespace {
OpxCreator<BatchNormOpx>
    batchNormOpxCreator({Onnx::Operators::BatchNormalization_6,
                         Onnx::Operators::BatchNormalization_7,
                         Onnx::Operators::BatchNormalization_9});
OpxCreator<BatchNormGradOpx>
    batchNormGradOpxCreator(Onnx::GradOperators::BatchNormalizationGrad);
} // namespace

} // namespace popx
} // namespace poponnx
