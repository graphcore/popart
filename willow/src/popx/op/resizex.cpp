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

  auto input  = getInTensor(ResizeOp::getInIndex());
  auto result = cloneNcopy(prog, input).getPoplarTensor();
  for (int i = 0; i < input.getPoplarTensor().rank(); i++) {
    if (resizeOp.getNearestMode() == ResizeNearestMode::Pytorch) {
      if (result.shape().at(i) != outShape.at(i)) {
        result =
            resizeDim(result, i, outShape.at(i), resizeOp.getScales().at(i));
      }
    } else {
      // Even if the output shape is the same, resize can still have an affect
      // on the values. Instead scale is checked.
      auto scale = resizeOp.getScales().at(i);
      if (scale != 1.0f) {
        result = resizeDim(result, i, outShape.at(i), scale);
      }
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

  auto nearestModeIsNot = [&](const std::vector<ResizeNearestMode> &xs) {
    auto nearestMode = getOp<ResizeOp>().getNearestMode();
    for (auto x : xs) {
      if (nearestMode == x) {
        return false;
      }
    }
    return true;
  };

  // Use poplar::Tensor::upsample if possible, as our generalised method for
  // float scales is extremely expensive on tensor expressions. If the scale is
  // a positive integer, and the resize mode is not floor or ceil, it is ok to
  // use poplars upsample. Poplars upsample works equally well for both resize
  // modes, round_prefer_floor, and round_prefer_ceil. If we look at the
  // equation used to transform the index:
  //   `rounding_mode((i + 0.5) / scale - 0.5)`
  // This will only return a difference result if the answer to
  //   `(i + 0.5) / scale - 0.5`
  // is `x.5`. If we then take the equation:
  //   `(i + 0.5) / scale - 0.5 = 0.5`
  // and rearrange it for s, we get:
  //   `s = i + 0.5`
  // but we know both s and i have to be integers and this can not be satisfied.
  if (scaleIsNonNegInt &&
      nearestModeIsNot({ResizeNearestMode::Floor, ResizeNearestMode::Ceil})) {
    return input.upsample(scale, dim, poplar::UpsampleMethod::REPEAT);
  } else {
    return resizeNearestNeighbour(input, dim, size, scale);
  }
}

namespace {

float round_prefer_floor(float x) {
  if (x - std::floor(x) <= 0.5f) {
    return std::floor(x);
  } else {
    return std::ceil(x);
  }
}

} // namespace

poplar::Tensor ResizeOpx::resizeNearestNeighbour(poplar::Tensor &input,
                                                 int dim,
                                                 int64_t size,
                                                 float scale) const {
  auto nearestMode    = getOp<ResizeOp>().getNearestMode();
  auto transformIndex = [nearestMode, scale](int i, int maxIndex) {
    auto onnxTransform = [scale](int i) {
      return (static_cast<float>(i) + 0.5f) / scale - 0.5f;
    };

    switch (nearestMode) {
    case ResizeNearestMode::RoundPreferCeil:
      return static_cast<int>(std::round(onnxTransform(i)));
    case ResizeNearestMode::RoundPreferFloor:
      return static_cast<int>(round_prefer_floor(onnxTransform(i)));
    case ResizeNearestMode::Floor:
      return std::max(static_cast<int>(std::floor(onnxTransform(i))), 0);
    case ResizeNearestMode::Ceil:
      return std::min(static_cast<int>(std::ceil(onnxTransform(i))), maxIndex);
    case ResizeNearestMode::Pytorch:
      return static_cast<int>(std::floor(static_cast<float>(i) / scale));
    default:
      throw error("Unrecognized ResizeNearestMode {}",
                  static_cast<int>(nearestMode));
    }
  };

  auto slices = split(input, dim);

  std::vector<poplar::Tensor> toConcat;
  for (int i = 0; i < size; i++) {
    int idx = transformIndex(i, slices.size() - 1);
    toConcat.push_back(slices.at(idx));
  }

  return poplar::concat(toConcat, dim);
}

ResizeGradOpx::ResizeGradOpx(Op *op, Devicex *devicex) : PopOpx(op, devicex) {
  verifyOp<ResizeGradOp>(op);
}

void ResizeGradOpx::grow(poplar::program::Sequence &prog) const {
  auto &op   = getOp<ResizeGradOp>();
  auto input = getInTensor(ResizeGradOp::getInIndex());

  auto inShape  = op.inShape(ResizeGradOp::getInIndex());
  auto outShape = op.outShape(ResizeGradOp::getOutIndex());

  auto result = cloneNcopy(prog, input).getPoplarTensor();
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
