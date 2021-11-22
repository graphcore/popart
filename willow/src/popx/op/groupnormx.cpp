// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <popart/error.hpp>
#include <popart/ir.hpp>
#include <popart/op/groupnorm.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/groupnormx.hpp>
#include <popart/popx/opxmanager.hpp>

#include <poplar/Tensor.hpp>
#include <poplin/Norms.hpp>
#include <popnn/GroupNorm.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Expr.hpp>

namespace poplar {
using Shape = std::vector<std::size_t>;
}

namespace pe = popops::expr;

namespace popart {
namespace popx {

GroupNormOpx::GroupNormOpx(Op *op, Devicex *devicex) : NormOpx(op, devicex) {
  verifyOp<GroupNormOp>(op, {Onnx::CustomOperators::GroupNormalization_1});
}

void GroupNormOpx::grow(snap::program::Sequence &prog) const {

  auto &op = getOp<GroupNormOp>();

  // Get the attributes
  float epsilon      = op.getEpsilon();
  int64_t num_groups = op.getNumGroups();
  // Check for stable algorithm session option.
  bool stable_algo = op.getIr().getSessionOptions().enableStableNorm;

  // int64_t num_channels = op.getNumChannels();

  // Get the inputs
  auto input = getInTensor(GroupNormOp::getXInIndex()).getPoplarTensor();
  auto scale = getInTensor(GroupNormOp::getScaleInIndex()).getPoplarTensor();
  auto b     = getInTensor(GroupNormOp::getBInIndex()).getPoplarTensor();

  // Calculate the mean and the inverse standard deviation
  poplar::Tensor mean;
  poplar::Tensor invStdDev;

  std::tie(mean, invStdDev) =
      popnn::gn::groupNormStatistics(graph().getPoplarGraph(),
                                     input,
                                     epsilon,
                                     prog.getPoplarSequence(),
                                     static_cast<unsigned int>(num_groups),
                                     false,
                                     stable_algo,
                                     poplar::FLOAT,
                                     debugContext("groupNormStatistics"));

  // Calculate the normalization
  auto result = popnn::gn::groupNormalise(graph().getPoplarGraph(),
                                          input,
                                          scale,
                                          b,
                                          mean,
                                          invStdDev,
                                          prog.getPoplarSequence(),
                                          debugContext("groupNorm"));

  // Return the result
  setOutTensor(GroupNormOp::getYOutIndex(),
               snap::Tensor{result.first, graph()});
  setOutTensor(GroupNormOp::getMeanOutIndex(), snap::Tensor{mean, graph()});
  setOutTensor(GroupNormOp::getInvStdDevOutIndex(),
               snap::Tensor{invStdDev, graph()});
}

GroupNormGradOpx::GroupNormGradOpx(Op *op, Devicex *devicex)
    : NormOpx(op, devicex) {
  verifyOp<GroupNormGradOp>(op, Onnx::GradOperators::GroupNormalizationGrad);
}

void GroupNormGradOpx::grow(snap::program::Sequence &prog) const {

  auto x = getInTensor(GroupNormGradOp::getXInIndex()).getPoplarTensor();
  auto yGrad =
      getInTensor(GroupNormGradOp::getYGradInIndex()).getPoplarTensor();
  auto scale =
      getInTensor(GroupNormGradOp::getScaleInIndex()).getPoplarTensor();
  auto mean = getInTensor(GroupNormGradOp::getMeanInIndex()).getPoplarTensor();
  auto invStdDev =
      getInTensor(GroupNormGradOp::getInvStdDevInIndex()).getPoplarTensor();

  poplar::Tensor xWhitened =
      popnn::gn::groupNormWhiten(graph().getPoplarGraph(),
                                 x,
                                 mean,
                                 invStdDev,
                                 prog.getPoplarSequence(),
                                 debugContext("whitenedActs"));

  // Compute the delta for the operand
  poplar::Tensor xGrad =
      popnn::gn::groupNormGradients(graph().getPoplarGraph(),
                                    xWhitened,
                                    yGrad,
                                    invStdDev,
                                    scale,
                                    prog.getPoplarSequence(),
                                    poplar::FLOAT,
                                    debugContext("operandGrad"));

  // Compute the deltas for scaled and offset
  poplar::Tensor scaleGrad;
  poplar::Tensor bGrad;
  std::tie(scaleGrad, bGrad) =
      popnn::gn::groupNormParamGradients(graph().getPoplarGraph(),
                                         xWhitened,
                                         yGrad,
                                         prog.getPoplarSequence(),
                                         poplar::FLOAT,
                                         debugContext("scaleOffsetGrads"));

  // Return the result
  setOutTensor(GroupNormGradOp::getXGradOutIndex(),
               snap::Tensor{xGrad, graph()});
  setOutTensor(GroupNormGradOp::getScaleOutIndex(),
               snap::Tensor{scaleGrad, graph()});
  setOutTensor(GroupNormGradOp::getBOutIndex(), snap::Tensor{bGrad, graph()});
}

namespace {
OpxCreator<GroupNormOpx>
    groupNormOpxCreator({Onnx::CustomOperators::GroupNormalization_1});
OpxCreator<GroupNormGradOpx>
    groupNormGradOpxCreator(Onnx::GradOperators::GroupNormalizationGrad);
} // namespace

} // namespace popx
} // namespace popart
