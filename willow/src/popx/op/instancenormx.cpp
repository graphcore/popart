#include <poponnx/error.hpp>
#include <poponnx/ir.hpp>
#include <poponnx/op/instancenorm.hpp>
#include <poponnx/popx/devicex.hpp>
#include <poponnx/popx/op/instancenormx.hpp>
#include <poponnx/popx/opxmanager.hpp>

#include <poplar/Tensor.hpp>
#include <poplin/Norms.hpp>
#include <popnn/InstanceNorm.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Expr.hpp>

namespace poplar {
using Shape = std::vector<std::size_t>;
}

namespace pe = popops::expr;

namespace poponnx {
namespace popx {

InstanceNormOpx::InstanceNormOpx(Op *op, Devicex *devicex)
    : NormOpx(op, devicex) {
  verifyOp<InstanceNormOp>(op, {Onnx::Operators::InstanceNormalization_6});
}

void InstanceNormOpx::grow(poplar::program::Sequence &prog) const {

  // Get the inputs
  auto input = get(inId(InstanceNormOp::getInputInIndex()));
  auto scale = get(inId(InstanceNormOp::getScaleInIndex()));
  auto b     = get(inId(InstanceNormOp::getBInIndex()));

  // Get the attributes
  float epsilon = getOp<InstanceNormOp>().getEpsilon();

  // Convert input shape to poplar rules
  poplar::Tensor inputP;
  poplar::Shape nonBroadcastDims;
  std::tie(inputP, nonBroadcastDims) = convertOnnxInputToPoplarInput(input);

  // Calculate the mean and the inverse standard deviation
  poplar::Tensor mean;
  poplar::Tensor invStdDev;
  std::tie(mean, invStdDev) =
      popnn::in::instanceNormStatistics(graph(), inputP, epsilon, prog, false);

  // Calculate the normalization
  auto result = popnn::in::instanceNormalise(graph(),
                                             input,
                                             scale,
                                             b,
                                             mean,
                                             invStdDev,
                                             prog,
                                             idStr() + "/instanceNorm");

  // Convert the output back into the input format
  poplar::Tensor y =
      convertPoplarOutputToOnnxOutput(result.first, nonBroadcastDims);

  // Return the result
  insert(outId(InstanceNormOp::getOutIndex()), y);
  insert(outId(InstanceNormOp::getMeanOutIndex()), mean);
  insert(outId(InstanceNormOp::getInvStdDevOutIndex()), invStdDev);
}

InstanceNormGradOpx::InstanceNormGradOpx(Op *op, Devicex *devicex)
    : NormOpx(op, devicex) {
  verifyOp<InstanceNormGradOp>(op,
                               Onnx::GradOperators::InstanceNormalizationGrad);
}

void InstanceNormGradOpx::grow(poplar::program::Sequence &prog) const {
  auto out_grad    = get(inId(InstanceNormGradOp::getOutGradInIndex()));
  auto input       = get(inId(InstanceNormGradOp::getInputInIndex()));
  auto scale       = get(inId(InstanceNormGradOp::getScaleInIndex()));
  auto mean        = get(inId(InstanceNormGradOp::getMeanInIndex()));
  auto inv_std_dev = get(inId(InstanceNormGradOp::getInvStdDevInIndex()));

  auto input_whitened = popnn::in::instanceNormWhiten(
      graph(), input, mean, inv_std_dev, prog, idStr());

  auto input_grad = popnn::in::instanceNormGradients(
      graph(),
      input_whitened,
      out_grad,
      inv_std_dev,
      scale,
      prog,
      poplar::FLOAT, // TODO: could this be HALF?
      idStr());

  poplar::Tensor scale_grad, b_grad;
  std::tie(scale_grad, b_grad) = popnn::in::instanceNormParamGradients(
      graph(), input_whitened, out_grad, prog, poplar::FLOAT, idStr());

  insert(outId(InstanceNormGradOp::getInputOutIndex()), input_grad);
  insert(outId(InstanceNormGradOp::getScaleOutIndex()), scale_grad);
  insert(outId(InstanceNormGradOp::getBOutIndex()), b_grad);
}

namespace {
OpxCreator<InstanceNormOpx>
    instanceNormOpxCreator({Onnx::Operators::InstanceNormalization_6});
OpxCreator<InstanceNormGradOpx>
    instanceNormGradOpxCreator(Onnx::GradOperators::InstanceNormalizationGrad);
} // namespace

} // namespace popx
} // namespace poponnx
