// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <cstddef>
#include <snap/Graph.hpp>
#include <snap/Program.hpp>
#include <snap/Tensor.hpp>
#include <tuple>
#include <utility>
#include <vector>
#include <poplar/Tensor.hpp>
#include <poplar/Type.hpp>
#include <popnn/InstanceNorm.hpp>
#include <popops/ExprOp.hpp>
#include <popart/ir.hpp>
#include <popart/op/instancenorm.hpp>
#include <popart/popx/op/instancenormx.hpp>
#include <popart/popx/op/normx.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/operators.hpp"
#include "popart/sessionoptions.hpp"

namespace popart {
class Op;
namespace popx {
class Devicex;
} // namespace popx
} // namespace popart

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

void InstanceNormOpx::grow(snap::program::Sequence &prog) const {

  auto &op = getOp<InstanceNormOp>();

  // Get the inputs
  auto input = getInTensor(InstanceNormOp::getInputInIndex()).getPoplarTensor();
  auto scale = getInTensor(InstanceNormOp::getScaleInIndex()).getPoplarTensor();
  auto b     = getInTensor(InstanceNormOp::getBInIndex()).getPoplarTensor();

  // Get the attributes
  float epsilon = getOp<InstanceNormOp>().getEpsilon();

  // Check for stable algorithm session option.
  bool stable_algo = op.getIr().getSessionOptions().enableStableNorm;

  // Calculate the mean and the inverse standard deviation
  poplar::Tensor mean;
  poplar::Tensor invStdDev;
  std::tie(mean, invStdDev) =
      popnn::in::instanceNormStatistics(graph().getPoplarGraph(),
                                        input,
                                        epsilon,
                                        prog.getPoplarSequence(),
                                        false,
                                        stable_algo,
                                        poplar::FLOAT,
                                        debugContext("instanceNormStatistics"));

  // Calculate the normalization
  auto result = popnn::in::instanceNormalise(graph().getPoplarGraph(),
                                             input,
                                             scale,
                                             b,
                                             mean,
                                             invStdDev,
                                             prog.getPoplarSequence(),
                                             debugContext("instanceNorm"));

  // Return the result
  setOutTensor(InstanceNormOp::getOutIndex(),
               snap::Tensor{result.first, graph()});
  setOutTensor(InstanceNormOp::getMeanOutIndex(), snap::Tensor{mean, graph()});
  setOutTensor(InstanceNormOp::getInvStdDevOutIndex(),
               snap::Tensor{invStdDev, graph()});
}

InstanceNormGradOpx::InstanceNormGradOpx(Op *op, Devicex *devicex)
    : NormOpx(op, devicex) {
  verifyOp<InstanceNormGradOp>(op,
                               Onnx::GradOperators::InstanceNormalizationGrad);
}

void InstanceNormGradOpx::grow(snap::program::Sequence &prog) const {
  auto out_grad =
      getInTensor(InstanceNormGradOp::getOutGradInIndex()).getPoplarTensor();
  auto input =
      getInTensor(InstanceNormGradOp::getInputInIndex()).getPoplarTensor();
  auto scale =
      getInTensor(InstanceNormGradOp::getScaleInIndex()).getPoplarTensor();
  auto mean =
      getInTensor(InstanceNormGradOp::getMeanInIndex()).getPoplarTensor();
  auto inv_std_dev =
      getInTensor(InstanceNormGradOp::getInvStdDevInIndex()).getPoplarTensor();

  auto input_whitened =
      popnn::in::instanceNormWhiten(graph().getPoplarGraph(),
                                    input,
                                    mean,
                                    inv_std_dev,
                                    prog.getPoplarSequence(),
                                    debugContext("instanceNormWhiten"));

  auto input_grad = popnn::in::instanceNormGradients(
      graph().getPoplarGraph(),
      input_whitened,
      out_grad,
      inv_std_dev,
      scale,
      prog.getPoplarSequence(),
      poplar::FLOAT, // TODO: could this be HALF?
      debugContext("instanceNormGradients"));

  poplar::Tensor scale_grad, b_grad;
  std::tie(scale_grad, b_grad) = popnn::in::instanceNormParamGradients(
      graph().getPoplarGraph(),
      input_whitened,
      out_grad,
      prog.getPoplarSequence(),
      poplar::FLOAT,
      debugContext("instanceNormParamGradients"));

  setOutTensor(InstanceNormGradOp::getInputOutIndex(),
               snap::Tensor{input_grad, graph()});
  setOutTensor(InstanceNormGradOp::getScaleOutIndex(),
               snap::Tensor{scale_grad, graph()});
  setOutTensor(InstanceNormGradOp::getBOutIndex(),
               snap::Tensor{b_grad, graph()});
}

namespace {
OpxCreator<InstanceNormOpx>
    instanceNormOpxCreator({Onnx::Operators::InstanceNormalization_6});
OpxCreator<InstanceNormGradOpx>
    instanceNormGradOpxCreator(Onnx::GradOperators::InstanceNormalizationGrad);
} // namespace

} // namespace popx
} // namespace popart
