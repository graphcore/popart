// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <popart/error.hpp>
#include <popart/ir.hpp>
#include <popart/op/instancenorm.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/instancenormx.hpp>
#include <popart/popx/opxmanager.hpp>

#include <poplar/Tensor.hpp>
#include <poplin/Norms.hpp>
#include <popnn/InstanceNorm.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Expr.hpp>

namespace poplar {
using Shape = std::vector<std::size_t>;
}

namespace pe = popops::expr;

namespace popart {
namespace popx {

InstanceNormOpx::InstanceNormOpx(Op *op, Devicex *devicex)
    : NormOpx(op, devicex) {
  verifyOp<InstanceNormOp>(op, {Onnx::Operators::InstanceNormalization_6});
}

void InstanceNormOpx::grow(poplar::program::Sequence &prog) const {

  auto &op = getOp<InstanceNormOp>();

  // Get the inputs
  auto input = getInTensor(InstanceNormOp::getInputInIndex());
  auto scale = getInTensor(InstanceNormOp::getScaleInIndex());
  auto b     = getInTensor(InstanceNormOp::getBInIndex());

  // Get the attributes
  float epsilon = getOp<InstanceNormOp>().getEpsilon();

  // Check for stable algorithm session option.
  bool stable_algo = op.getIr().getSessionOptions().enableStableNorm;

  // Calculate the mean and the inverse standard deviation
  poplar::Tensor mean;
  poplar::Tensor invStdDev;
  std::tie(mean, invStdDev) =
      popnn::in::instanceNormStatistics(graph(),
                                        input,
                                        epsilon,
                                        prog,
                                        false,
                                        stable_algo,
                                        poplar::FLOAT,
                                        debugContext("instanceNormStatistics"));

  // Calculate the normalization
  auto result = popnn::in::instanceNormalise(graph(),
                                             input,
                                             scale,
                                             b,
                                             mean,
                                             invStdDev,
                                             prog,
                                             debugContext("instanceNorm"));

  // Return the result
  setOutTensor(InstanceNormOp::getOutIndex(), result.first);
  setOutTensor(InstanceNormOp::getMeanOutIndex(), mean);
  setOutTensor(InstanceNormOp::getInvStdDevOutIndex(), invStdDev);
}

InstanceNormGradOpx::InstanceNormGradOpx(Op *op, Devicex *devicex)
    : NormOpx(op, devicex) {
  verifyOp<InstanceNormGradOp>(op,
                               Onnx::GradOperators::InstanceNormalizationGrad);
}

void InstanceNormGradOpx::grow(poplar::program::Sequence &prog) const {
  auto out_grad    = getInTensor(InstanceNormGradOp::getOutGradInIndex());
  auto input       = getInTensor(InstanceNormGradOp::getInputInIndex());
  auto scale       = getInTensor(InstanceNormGradOp::getScaleInIndex());
  auto mean        = getInTensor(InstanceNormGradOp::getMeanInIndex());
  auto inv_std_dev = getInTensor(InstanceNormGradOp::getInvStdDevInIndex());

  auto input_whitened =
      popnn::in::instanceNormWhiten(graph(),
                                    input,
                                    mean,
                                    inv_std_dev,
                                    prog,
                                    debugContext("instanceNormWhiten"));

  auto input_grad = popnn::in::instanceNormGradients(
      graph(),
      input_whitened,
      out_grad,
      inv_std_dev,
      scale,
      prog,
      poplar::FLOAT, // TODO: could this be HALF?
      debugContext("instanceNormGradients"));

  poplar::Tensor scale_grad, b_grad;
  std::tie(scale_grad, b_grad) = popnn::in::instanceNormParamGradients(
      graph(),
      input_whitened,
      out_grad,
      prog,
      poplar::FLOAT,
      debugContext("instanceNormParamGradients"));

  setOutTensor(InstanceNormGradOp::getInputOutIndex(), input_grad);
  setOutTensor(InstanceNormGradOp::getScaleOutIndex(), scale_grad);
  setOutTensor(InstanceNormGradOp::getBOutIndex(), b_grad);
}

namespace {
OpxCreator<InstanceNormOpx>
    instanceNormOpxCreator({Onnx::Operators::InstanceNormalization_6});
OpxCreator<InstanceNormGradOpx>
    instanceNormGradOpxCreator(Onnx::GradOperators::InstanceNormalizationGrad);
} // namespace

} // namespace popx
} // namespace popart
