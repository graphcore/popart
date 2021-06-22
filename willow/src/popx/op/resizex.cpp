#include <cmath>

#include <popops/ElementWise.hpp>
#include <popops/Zero.hpp>

#include <popart/logging.hpp>
#include <popart/op/resize.hpp>
#include <popart/popx/op/resizex.hpp>
#include <popart/popx/opxmanager.hpp>

namespace popart {
namespace popx {

ResizeOpx::ResizeOpx(Op *op, Devicex *devicex) : PopOpx(op, devicex) {
  verifyOp<ResizeOp>(op);
}

void ResizeOpx::grow(poplar::program::Sequence &prog) const {
  auto &resizeOp = getOp<ResizeOp>();
  auto outShape  = resizeOp.outShape(ResizeOp::getOutIndex());

  auto input  = getInTensor(ResizeOp::getInIndex()).getPoplarTensor();
  auto result = cloneNcopy(prog, input);
  for (int i = 0; i < input.rank(); i++) {
    if (result.shape().at(i) != outShape.at(i)) {
      result = resizeDim(result, i, outShape.at(i), resizeOp.getScales().at(i));
    }
  }

  setOutTensor(ResizeOp::getOutIndex(), snap::Tensor{result, graph()});
}

namespace {

std::vector<poplar::Tensor> split(const poplar::Tensor &input, int dim) {
  std::vector<poplar::Tensor> result;
  for (int i = 0; i < input.dim(dim); i++) {
    result.push_back(input.slice(i, i + 1, dim));
  }
  return result;
}

poplar::Tensor nativeNonNegIntegerResize(poplar::Tensor &input,
                                         const int dim,
                                         const float scale) {
  return input.upsample(scale, dim, poplar::UpsampleMethod::REPEAT);
}

} // namespace

poplar::Tensor ResizeOpx::resizeDim(poplar::Tensor &input,
                                    int dim,
                                    int64_t size,
                                    float scale) const {
  // Check float is an int.
  // https://stackoverflow.com/a/25274904
  constexpr float eps      = 0.00001f;
  const float roundedScale = std::roundf(scale);
  const bool scaleIsNonNegInt =
      std::fabs(roundedScale - scale) <= eps && roundedScale >= 0.f;

  // Use the native resize method where possible, as our generalised method for
  // float scales is extremely expensive on tensor expressions.

  return scaleIsNonNegInt ? nativeNonNegIntegerResize(input, dim, scale)
                          : resizeNearestNeighbour(input, dim, size, scale);
}

poplar::Tensor ResizeOpx::resizeNearestNeighbour(poplar::Tensor &input,
                                                 int dim,
                                                 int64_t size,
                                                 float scale) const {
  auto slices = split(input, dim);

  std::vector<poplar::Tensor> toConcat;
  for (int i = 0; i < size; i++) {
    int idx = static_cast<int>(std::floor(static_cast<float>(i) / scale));
    toConcat.push_back(slices.at(idx));
  }

  return poplar::concat(toConcat, dim);
}

ResizeGradOpx::ResizeGradOpx(Op *op, Devicex *devicex) : PopOpx(op, devicex) {
  verifyOp<ResizeGradOp>(op);
}

void ResizeGradOpx::grow(poplar::program::Sequence &prog) const {
  auto &op   = getOp<ResizeGradOp>();
  auto input = getInTensor(ResizeGradOp::getInIndex()).getPoplarTensor();

  auto inShape  = op.inShape(ResizeGradOp::getInIndex());
  auto outShape = op.outShape(ResizeGradOp::getOutIndex());

  auto result = cloneNcopy(prog, input);
  for (int dimension = 0; dimension < inShape.size(); dimension++) {
    auto inDim  = inShape.at(dimension);
    auto outDim = outShape.at(dimension);
    if (inDim > outDim) {
      result = reduceDimension(
          prog, result, dimension, 1.0f / op.getFwdScales().at(dimension));
    } else if (inDim < outDim) {
      result = padDimension(prog,
                            result,
                            dimension,
                            outDim,
                            1.0f / op.getFwdScales().at(dimension));
    }
  }

  setOutTensor(ResizeGradOp::getOutIndex(), snap::Tensor{result, graph()});
}

poplar::Tensor ResizeGradOpx::reduceDimension(poplar::program::Sequence &prog,
                                              const poplar::Tensor &input,
                                              int dimension,
                                              float scale) const {
  auto slices = split(input, dimension);

  std::map<int, poplar::Tensor> resultMap;
  for (int i = 0; i < slices.size(); i++) {
    int idx = static_cast<int>(std::floor(i * scale));
    if (resultMap.find(idx) == resultMap.end()) {
      resultMap[idx] = slices[i];
    } else {
      resultMap[idx] = popops::map(graph().getPoplarGraph(),
                                   popops::expr::BinaryOpType::ADD,
                                   resultMap[idx],
                                   slices[i],
                                   prog);
    }
  }

  std::vector<poplar::Tensor> toConcat;
  for (int i = 0; i < resultMap.size(); i++) {
    toConcat.push_back(resultMap.at(i));
  }
  return poplar::concat(toConcat, dimension);
}

poplar::Tensor ResizeGradOpx::padDimension(poplar::program::Sequence &prog,
                                           const poplar::Tensor &input,
                                           int dimension,
                                           int64_t newSize,
                                           float scale) const {
  auto slices        = split(input, dimension);
  auto paddingTensor = graph().getPoplarGraph().addVariable(
      input.elementType(), slices.at(0).shape());
  popops::zero(graph().getPoplarGraph(),
               paddingTensor,
               prog,
               debugContext("zeroPadding"));

  std::vector<poplar::Tensor> toConcat(newSize, paddingTensor);
  for (int i = 0; i < slices.size(); i++) {
    int idx          = static_cast<int>(std::floor(i * scale));
    toConcat.at(idx) = slices.at(i);
  }

  return poplar::concat(toConcat, dimension);
}

namespace {
OpxCreator<ResizeOpx> resizeOpxCreator(Onnx::CustomOperators::Resize);
OpxCreator<ResizeGradOpx> resizeGradOpxCreator(Onnx::GradOperators::ResizeGrad);
} // namespace

} // namespace popx
} // namespace popart
