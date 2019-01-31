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

InstanceNormOpx::InstanceNormOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
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
}

namespace {
OpxCreator<InstanceNormOpx>
    instanceNormOpxCreator({Onnx::Operators::InstanceNormalization_6});
} // namespace

} // namespace popx
} // namespace poponnx
