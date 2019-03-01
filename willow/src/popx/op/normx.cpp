#include <poponnx/error.hpp>
#include <poponnx/ir.hpp>
#include <poponnx/op.hpp>
#include <poponnx/popx/devicex.hpp>
#include <poponnx/popx/op/normx.hpp>
#include <poponnx/popx/opxmanager.hpp>

#include <poplar/Tensor.hpp>
#include <poplin/Norms.hpp>
#include <popnn/GroupNorm.hpp>
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
std::pair<poplar::Tensor, poplar::Shape>
NormOpx::convertOnnxInputToPoplarInput(const poplar::Tensor &onnxInput) const {
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
poplar::Tensor NormOpx::convertPoplarOutputToOnnxOutput(
    const poplar::Tensor &poplarOutput,
    const poplar::Shape &nonBroadcastDimensions) const {
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

// convert inverse standard deviation to variance
poplar::Tensor NormOpx::convertInvSdToVar(poplar::program::Sequence &prog,
                                          const poplar::Tensor &invSd,
                                          float epsilon) const {

  popops::mapInPlace(graph(),
                     pe::InvStdDevToVariance(pe::_1, pe::Const(epsilon)),
                     {invSd},
                     prog,
                     idStr() + "/invSdToVar");
  return invSd;
}

// convert variant to inverse standard deviation
poplar::Tensor NormOpx::convertVarToInvSd(poplar::program::Sequence &prog,
                                          const poplar::Tensor &var,
                                          float epsilon) const {
  popops::mapInPlace(graph(),
                     pe::VarianceToInvStdDev(pe::_1, pe::Const(epsilon)),
                     {var},
                     prog,
                     idStr() + "/varToInvSd");
  return var;
}

NormOpx::NormOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {}

} // namespace popx
} // namespace poponnx
