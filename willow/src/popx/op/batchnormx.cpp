// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <poprithms/logging/timepartitionlogger.hpp>
#include <popart/error.hpp>
#include <popart/ir.hpp>
#include <popart/op/batchnorm.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/batchnormx.hpp>
#include <popart/popx/opxmanager.hpp>

#include <poplar/Tensor.hpp>
#include <poplin/Norms.hpp>
#include <popnn/BatchNorm.hpp>
#include <popops/Cast.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Expr.hpp>

#include <cmath>

namespace poplar {
using Shape = std::vector<std::size_t>;
}

namespace pe = popops::expr;

namespace popart {
namespace popx {

BatchNormOpx::BatchNormOpx(Op *op, Devicex *devicex) : NormOpx(op, devicex) {
  verifyOp<BatchNormOp>(op,
                        {Onnx::Operators::BatchNormalization_6,
                         Onnx::Operators::BatchNormalization_7,
                         Onnx::Operators::BatchNormalization_9});
}

// Not clear to me if batchNormalise is meant update the mean/var then how can
// they be constant tensors
snap::Tensor BatchNormOpx::batchNormalise(poplar::program::Sequence &prog,
                                          const snap::Tensor &x,
                                          const snap::Tensor &scale,
                                          const snap::Tensor &b,
                                          const snap::Tensor &mean,
                                          const snap::Tensor &invSd) const {

  //  combinedMultiplicand = gamma / sDev
  //                       = gamma * invSd
  auto multiplcand =
      popops::map(graph().getPoplarGraph(),
                  pe::Mul(pe::_1, pe::_2),
                  {scale.getPoplarTensor(), invSd.getPoplarTensor()},
                  prog,
                  debugContext("multiplicand"));

  // addend = beta - gamma * mean / sdDev
  //        = beta - gamma * mean * invSd
  auto addend =
      popops::map(graph().getPoplarGraph(),
                  pe::Sub(pe::_1, pe::Mul(pe::_2, pe::_3)),
                  {b.getPoplarTensor(), multiplcand, mean.getPoplarTensor()},
                  prog,
                  debugContext("addend"));

  // Perform the batchNorm
  return snap::Tensor{popnn::bn::batchNormalise(graph().getPoplarGraph(),
                                                x.getPoplarTensor(),
                                                multiplcand,
                                                addend,
                                                prog,
                                                debugContext("batchNormalise")),
                      graph()};
}

static bool isZeroElementArray(const poplar::Shape &shape) {
  return std::all_of(
      shape.begin(), shape.end(), [](int dim) -> bool { return dim == 0; });
}

void BatchNormOpx::grow(poplar::program::Sequence &prog) const {

  const auto growTimeTracker =
      op_p->getIr().timePartitionLogger().scopedStopwatch(
          "Lowering BatchNorm to Poplar (\"grow\")");

  auto &op = getOp<BatchNormOp>();

  // Using the input names as per the onnx spec.
  auto x     = getInTensor(BatchNormOp::getXInIndex());
  auto scale = getInTensor(BatchNormOp::getScaleInIndex());
  auto b     = getInTensor(BatchNormOp::getBInIndex());
  auto mean  = getInTensor(BatchNormOp::getMeanInIndex());
  auto var   = getInTensor(BatchNormOp::getVarInIndex());

  // Variables to store the desired output shapes in case of spatial=False.
  std::vector<size_t> yShape            = x.getPoplarTensor().shape();
  std::vector<size_t> otherOutputsShape = scale.getPoplarTensor().shape();

  if (!op.getSpatial()) {
    // If spatial=False we must normalise every feature separately. We achieve
    // this with by transforming the inputs, running batchnorm as if
    // spatial=True and then transforming the output back again, e.g.:
    //
    // - Transforming input x from [N, C, D1, ..., Dn] to [N, CxD1x...xDn, 1].
    // - Transforming inputs scale, b, mean, var from [C, D1, ..., Dn] to
    // [CxD1x...xDn] (if available)
    // - Applying batch normalization (with spatial=True) on the transformed
    // inputs.
    // - Transforming output y from [N, CxD1x...xDn, 1] to [N, C, D1, ..., Dn].
    // - Transforming outputs runningMean, runningVar, batchMean, batchVar from
    // [CxD1x...xDn] to [C, D1, ..., Dn] (if available)

    const size_t NUM_FEATURES =
        x.getPoplarTensor().numElements() / x.getPoplarTensor().dim(0);

    // Reshape the inputs.
    x = snap::Tensor{x.getPoplarTensor().reshape(
                         {x.getPoplarTensor().dim(0), NUM_FEATURES, 1}),
                     graph()};
    scale =
        snap::Tensor{scale.getPoplarTensor().reshape({NUM_FEATURES}), graph()};
    b = snap::Tensor{b.getPoplarTensor().reshape({NUM_FEATURES}), graph()};
    mean =
        snap::Tensor{mean.getPoplarTensor().reshape({NUM_FEATURES}), graph()};
    var = snap::Tensor{var.getPoplarTensor().reshape({NUM_FEATURES}), graph()};
  }

  // Lower the batch normalization operator for spatial=True.
  auto outputs = growSpatial(prog, op, x, scale, b, mean, var);

  if (!op.getSpatial()) {
    // As spatial=False we must transform the outputs back to their expected
    // form (see above). Note that some outputs are optional and depend on the
    // use case.
    outputs.y =
        snap::Tensor{outputs.y.getPoplarTensor().reshape(yShape), graph()};

    if (outputs.mean)
      outputs.mean = snap::Tensor{
          outputs.mean->getPoplarTensor().reshape(otherOutputsShape), graph()};
    if (outputs.var)
      outputs.var = snap::Tensor{
          outputs.var->getPoplarTensor().reshape(otherOutputsShape), graph()};
    if (outputs.savedMean)
      outputs.savedMean = snap::Tensor{
          outputs.savedMean->getPoplarTensor().reshape(otherOutputsShape),
          graph()};
    if (outputs.savedVar)
      outputs.savedVar = snap::Tensor{
          outputs.savedVar->getPoplarTensor().reshape(otherOutputsShape),
          graph()};
  }

  // Now we need to set the output tensors, where available.
  setOutTensor(BatchNormOp::getYOutIndex(), outputs.y);

  if (outputs.mean)
    setOutTensor(BatchNormOp::getMeanOutIndex(), *outputs.mean);
  if (outputs.var)
    setOutTensor(BatchNormOp::getVarOutIndex(), *outputs.var);
  if (outputs.savedMean)
    setOutTensor(BatchNormOp::getSavedMeanOutIndex(), *outputs.savedMean);
  if (outputs.savedVar)
    setOutTensor(BatchNormOp::getSavedVarOutIndex(), *outputs.savedVar);
}

BatchNormOpx::GrowSpatialOutput
BatchNormOpx::growSpatial(poplar::program::Sequence &prog,
                          BatchNormOp &op,
                          snap::Tensor &x,
                          snap::Tensor &scale,
                          snap::Tensor &b,
                          snap::Tensor &mean,
                          snap::Tensor &var) const {
  // OK. This is found more by trial and error. It appears that pytorch, is
  // using an unbiased running variance But the other question is what should we
  // do for onnx that does not state if the running_variance is biased or
  // unbiased. The ONNX calculation for running_xxx is different than pyTorch
  GrowSpatialOutput result;

  // Using the attribute names as per the onnx spec.
  float epsilon  = op.getEpsilon();
  float momentum = op.getMomentum();

  // Check for stable algorithm session option.
  bool stable_algo = op.getIr().getSessionOptions().enableStableNorm;

  if (op.isTraining()) {

    // Special case - zero sized array
    if (isZeroElementArray(x.getPoplarTensor().shape())) {
      auto y         = snap::Tensor{graph().getPoplarGraph().addConstant(
                                x.getPoplarTensor().elementType(),
                                x.getPoplarTensor().shape(),
                                0,
                                debugContext("y")),
                            graph()};
      auto batchMean = snap::Tensor{graph().getPoplarGraph().addConstant(
                                        mean.getPoplarTensor().elementType(),
                                        {1},
                                        NAN,
                                        debugContext("mean")),
                                    graph()};
      auto batchVar  = snap::Tensor{graph().getPoplarGraph().addConstant(
                                       var.getPoplarTensor().elementType(),
                                       {1},
                                       NAN,
                                       debugContext("var")),
                                   graph()};
      graph().getPoplarGraph().setTileMapping(y.getPoplarTensor(), 0);
      graph().getPoplarGraph().setTileMapping(batchMean.getPoplarTensor(), 0);
      graph().getPoplarGraph().setTileMapping(batchVar.getPoplarTensor(), 0);

      result = GrowSpatialOutput({y,
                                  batchMean,
                                  batchVar,
                                  nonstd::optional<snap::Tensor>(),
                                  nonstd::optional<snap::Tensor>()});
    } else {
      poplar::Tensor batchMeanP, invSdP;
      std::tie(batchMeanP, invSdP) =
          popnn::bn::batchNormStatistics(graph().getPoplarGraph(),
                                         x.getPoplarTensor(),
                                         epsilon,
                                         prog,
                                         false,
                                         stable_algo,
                                         poplar::FLOAT,
                                         debugContext("normStats"));
      auto batchMean = snap::Tensor{batchMeanP, graph()};
      auto invSd     = snap::Tensor{invSdP, graph()};

      // batch normalise
      auto y = batchNormalise(prog, x, scale, b, batchMean, invSd);

      // convert updated inverse standard deviation back to variance.

      /*
      // Left this code in for now which computes the variance used in pytorch
      {
        // First we have to convert the invSd to the unbiased version
        const auto numElements = x.numElements() / x.dim(1);
        invSd = popops::map(graph().getPoplarGraph(),
                                  pe::Mul(pe::_1,
      pe::Sqrt(pe::Divide(pe::Const(numElements -1),pe::Const(numElements)))),
                                  {invSd},
                                  prog,
                                  idStr() + "/unbiasedInvSd");
      }
      */

      // Ensure batch mean is the same type as mean so that running mean can
      // be calculated
      if (batchMean.getPoplarTensor().elementType() !=
          mean.getPoplarTensor().elementType()) {
        batchMean =
            snap::Tensor{popops::cast(graph().getPoplarGraph(),
                                      batchMean.getPoplarTensor(),
                                      mean.getPoplarTensor().elementType(),
                                      prog,
                                      debugContext("cast_batchMean")),
                         graph()};
      }

      // Then convert the invSd to the variance
      auto batchVar = convertInvSdToVar(
          prog, invSd, epsilon, var.getPoplarTensor().elementType());

      // Calculate the running mean
      auto runningMean = snap::Tensor{
          popops::map(
              graph().getPoplarGraph(),
              pe::Add(
                  pe::Mul(pe::Sub(pe::Const(1), pe::Const(momentum)), pe::_2),
                  pe::Mul(pe::Const(momentum), pe::_1)),
              {mean.getPoplarTensor(), batchMean.getPoplarTensor()},
              prog,
              debugContext("runningMean")),
          graph()};

      // Calculate the running variance using the unbiased results
      auto runningVar = snap::Tensor{
          popops::map(
              graph().getPoplarGraph(),
              pe::Add(
                  pe::Mul(pe::Sub(pe::Const(1), pe::Const(momentum)), pe::_2),
                  pe::Mul(pe::Const(momentum), pe::_1)),
              {var.getPoplarTensor(), batchVar.getPoplarTensor()},
              prog,
              debugContext("runningVar")),
          graph()};

      // return the results
      result =
          GrowSpatialOutput({y, runningMean, runningVar, batchMean, batchVar});
    }
  } else {
    // When testing

    // Special case - zero sized array
    if (isZeroElementArray(x.getPoplarTensor().shape())) {
      auto y = snap::Tensor{graph().getPoplarGraph().addConstant(
                                x.getPoplarTensor().elementType(),
                                x.getPoplarTensor().shape(),
                                0,
                                debugContext("y")),
                            graph()};
      graph().getPoplarGraph().setTileMapping(y.getPoplarTensor(), 0);

      result = GrowSpatialOutput({y,
                                  nonstd::optional<snap::Tensor>(),
                                  nonstd::optional<snap::Tensor>(),
                                  nonstd::optional<snap::Tensor>(),
                                  nonstd::optional<snap::Tensor>()});
    } else {
      // convert variant to inverse standard deviation
      auto invSd = convertVarToInvSd(
          prog, var, epsilon, x.getPoplarTensor().elementType());

      // mean might have a different type so cast is required before
      // batchNormalise calculation
      if (mean.getPoplarTensor().elementType() !=
          x.getPoplarTensor().elementType()) {
        mean = snap::Tensor{popops::cast(graph().getPoplarGraph(),
                                         mean.getPoplarTensor(),
                                         x.getPoplarTensor().elementType(),
                                         prog,
                                         debugContext("cast_mean")),
                            graph()};
      }

      // batchnorm
      auto y = batchNormalise(prog, x, scale, b, mean, invSd);

      // return the result
      result = GrowSpatialOutput({y,
                                  nonstd::optional<snap::Tensor>(),
                                  nonstd::optional<snap::Tensor>(),
                                  nonstd::optional<snap::Tensor>(),
                                  nonstd::optional<snap::Tensor>()});
    }
  }

  return result;
}

std::tuple<snap::Tensor, snap::Tensor, snap::Tensor>
BatchNormGradOpx::batchNormaliseGrad(poplar::program::Sequence &prog,
                                     const snap::Tensor &x,
                                     const snap::Tensor &scale,
                                     const snap::Tensor &mean,
                                     const snap::Tensor &invSd,
                                     const snap::Tensor &yGrad) const {

  snap::Tensor xGrad, scaleGrad, bGrad;

  snap::Tensor xWhitened =
      snap::Tensor{popnn::bn::batchNormWhiten(graph().getPoplarGraph(),
                                              x.getPoplarTensor(),
                                              mean.getPoplarTensor(),
                                              invSd.getPoplarTensor(),
                                              prog,
                                              debugContext("WhitenedActs")),
                   graph()};

  // Compute the delta for the operand
  xGrad =
      snap::Tensor{popnn::bn::batchNormGradients(graph().getPoplarGraph(),
                                                 xWhitened.getPoplarTensor(),
                                                 yGrad.getPoplarTensor(),
                                                 invSd.getPoplarTensor(),
                                                 scale.getPoplarTensor(),
                                                 prog,
                                                 poplar::FLOAT,
                                                 debugContext("operandGrad")),
                   graph()};

  // Compute the deltas for scaled and offset
  poplar::Tensor scaleGradP, bGradP;
  std::tie(scaleGradP, bGradP) =
      popnn::bn::batchNormParamGradients(graph().getPoplarGraph(),
                                         xWhitened.getPoplarTensor(),
                                         yGrad.getPoplarTensor(),
                                         prog,
                                         poplar::FLOAT,
                                         debugContext("scaleOffsetGrads"));
  scaleGrad = snap::Tensor{scaleGradP, graph()};
  bGrad     = snap::Tensor{bGradP, graph()};

  return std::make_tuple(xGrad, scaleGrad, bGrad);
}

BatchNormGradOpx::BatchNormGradOpx(Op *op, Devicex *devicex)
    : NormOpx(op, devicex) {
  verifyOp<BatchNormGradOp>(op, Onnx::GradOperators::BatchNormalizationGrad);
}

void BatchNormGradOpx::grow(poplar::program::Sequence &prog) const {

  auto &op = getOp<BatchNormGradOp>();

  // Inputs
  auto x     = getInTensor(BatchNormGradOp::getXInIndex());
  auto scale = getInTensor(BatchNormGradOp::getScaleInIndex());
  auto mean  = getInTensor(BatchNormGradOp::getMeanInIndex());
  auto var   = getInTensor(BatchNormGradOp::getVarInIndex());
  auto yGrad = getInTensor(BatchNormGradOp::getYGradInIndex());

  // Variables to store the desired output shapes in case of spatial=False.
  std::vector<size_t> xShape            = yGrad.getPoplarTensor().shape();
  std::vector<size_t> otherOutputsShape = scale.getPoplarTensor().shape();

  if (!op.getSpatial()) {
    // If spatial=False we must do some reshaping here (akin to BatchNormOpx).
    const size_t NUM_FEATURES =
        x.getPoplarTensor().numElements() / x.getPoplarTensor().dim(0);

    // Reshape the inputs.
    x = snap::Tensor{x.getPoplarTensor().reshape(
                         {x.getPoplarTensor().dim(0), NUM_FEATURES, 1}),
                     graph()};
    scale =
        snap::Tensor{scale.getPoplarTensor().reshape({NUM_FEATURES}), graph()};
    mean =
        snap::Tensor{mean.getPoplarTensor().reshape({NUM_FEATURES}), graph()};
    var = snap::Tensor{var.getPoplarTensor().reshape({NUM_FEATURES}), graph()};
    yGrad = snap::Tensor{yGrad.getPoplarTensor().reshape(
                             {x.getPoplarTensor().dim(0), NUM_FEATURES, 1}),
                         graph()};
  }

  auto outputs = growSpatial(prog, op, x, scale, mean, var, yGrad);

  if (!op.getSpatial()) {
    // As spatial=False we must undo the reshaping here (aking to BatchNormOpx).
    outputs.xGrad =
        snap::Tensor{outputs.xGrad.getPoplarTensor().reshape(xShape), graph()};
    outputs.scaleGrad = snap::Tensor{
        outputs.scaleGrad.getPoplarTensor().reshape(otherOutputsShape),
        graph()};
    outputs.bGrad = snap::Tensor{
        outputs.bGrad.getPoplarTensor().reshape(otherOutputsShape), graph()};
  }

  setOutTensor(BatchNormGradOp::getXOutIndex(), outputs.xGrad);
  setOutTensor(BatchNormGradOp::getScaleOutIndex(), outputs.scaleGrad);
  setOutTensor(BatchNormGradOp::getBOutIndex(), outputs.bGrad);
}

BatchNormGradOpx::GrowSpatialOutput
BatchNormGradOpx::growSpatial(poplar::program::Sequence &prog,
                              BatchNormGradOp &op,
                              snap::Tensor &x,
                              snap::Tensor &scale,
                              snap::Tensor &mean,
                              snap::Tensor &var,
                              snap::Tensor &yGrad) const {
  GrowSpatialOutput result;

  // Attributes
  float epsilon = op.getEpsilon();

  // Special case - zero sized array
  if (isZeroElementArray(x.getPoplarTensor().shape())) {
    auto xGrad = snap::Tensor{
        graph().getPoplarGraph().addConstant(x.getPoplarTensor().elementType(),
                                             x.getPoplarTensor().shape(),
                                             0,
                                             debugContext("xGrad")),
        graph()};
    auto scaleGrad = snap::Tensor{
        graph().getPoplarGraph().addConstant(x.getPoplarTensor().elementType(),
                                             {1},
                                             0,
                                             debugContext("scaleGrad")),
        graph()};
    auto bGrad = snap::Tensor{
        graph().getPoplarGraph().addConstant(
            x.getPoplarTensor().elementType(), {1}, 0, debugContext("bGrad")),
        graph()};
    graph().getPoplarGraph().setTileMapping(xGrad.getPoplarTensor(), 0);
    graph().getPoplarGraph().setTileMapping(scaleGrad.getPoplarTensor(), 0);
    graph().getPoplarGraph().setTileMapping(bGrad.getPoplarTensor(), 0);
    result.xGrad     = xGrad;
    result.scaleGrad = scaleGrad;
    result.bGrad     = bGrad;
  } else {
    auto invSd = convertVarToInvSd(
        prog, var, epsilon, x.getPoplarTensor().elementType());

    // mean might have a different type so cast is required before
    // batchNormaliseGrad calculation
    if (mean.getPoplarTensor().elementType() !=
        x.getPoplarTensor().elementType()) {
      mean = snap::Tensor{popops::cast(graph().getPoplarGraph(),
                                       mean.getPoplarTensor(),
                                       x.getPoplarTensor().elementType(),
                                       prog,
                                       debugContext("cast_mean")),
                          graph()};
    }

    // batchnormgrad
    snap::Tensor xGrad, scaleGrad, bGrad;
    std::tie(xGrad, scaleGrad, bGrad) =
        batchNormaliseGrad(prog, x, scale, mean, invSd, yGrad);

    // return the results
    result.xGrad     = xGrad;
    result.scaleGrad = scaleGrad;
    result.bGrad     = bGrad;
  }

  return result;
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
} // namespace popart
