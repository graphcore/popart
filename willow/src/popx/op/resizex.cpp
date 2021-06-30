#include <cmath>

#include <popops/ElementWise.hpp>
#include <popops/Zero.hpp>

#include <popart/logging.hpp>
#include <popart/op/resize.hpp>
#include <popart/popx/op/resizex.hpp>
#include <popart/popx/opxmanager.hpp>

namespace popart {
namespace popx {

namespace {

struct ResizeParams {
  Shape inShape;
  Shape outShape;
  std::vector<float> scales;
  ResizeMode mode;
  ResizeNearestMode nearestMode;
  ResizeCoordinateTransformationMode coordinateTransformationMode;
};

ResizeParams getResizeParams(const ResizeOp &x) {
  return {x.inShape(ResizeOp::getInIndex()),
          x.outShape(ResizeOp::getOutIndex()),
          x.getScales(),
          x.getMode(),
          x.getNearestMode(),
          x.getCoordinateTransformationMode()};
}

std::vector<poplar::Tensor> split(const poplar::Tensor &input, int dim) {
  std::vector<poplar::Tensor> result;
  for (int i = 0; i < input.dim(dim); i++) {
    result.push_back(input.slice(i, i + 1, dim));
  }
  return result;
}

float round_prefer_floor(float x) {
  if (x - std::floor(x) <= 0.5f) {
    return std::floor(x);
  } else {
    return std::ceil(x);
  }
}

int clamp(int v, int lo, int hi) { return std::min(std::max(v, lo), hi); }

float coordinateTransformation(float idx, int dim, const ResizeParams &params) {
  auto mode   = params.coordinateTransformationMode;
  float scale = params.scales.at(dim);

  if (params.nearestMode == ResizeNearestMode::Pytorch) {
    return idx / scale;
  } else {

    switch (mode) {
    case ResizeCoordinateTransformationMode::HalfPixel:
      return (idx + 0.5f) / scale - 0.5f;
    case ResizeCoordinateTransformationMode::PytorchHalfPixel: {
      int inputSize = params.inShape.at(dim);
      float size    = static_cast<float>(inputSize) * scale;
      if (size > 1.0f) {
        return (idx + 0.5f) / scale - 0.5f;
      } else {
        return 0.0f;
      }
    }
    case ResizeCoordinateTransformationMode::Asymmetric:
      return idx / scale;
    case ResizeCoordinateTransformationMode::AlignCorners: {
      int inputSize = params.inShape.at(dim);
      float size    = static_cast<float>(inputSize) * scale;
      // (size - 1) results in divide by zero and a NAN.
      if (size == 1) {
        return 0;
      } else {
        return idx * (inputSize - 1) / (size - 1);
      }
    }
    default:
      throw error("Unsupported coordinate transformation mode");
    }
  }
}

int64_t applyNearestMode(float idx, const ResizeParams &params) {
  switch (params.nearestMode) {
  case ResizeNearestMode::RoundPreferCeil:
    return std::round(idx);
  case ResizeNearestMode::RoundPreferFloor:
    return round_prefer_floor(idx);
  case ResizeNearestMode::Floor:
    return std::floor(idx);
  case ResizeNearestMode::Ceil:
    return std::ceil(idx);
  case ResizeNearestMode::Pytorch:
    return std::floor(idx);
  default:
    throw error("Unrecognized ResizeNearestMode {}",
                static_cast<int>(params.nearestMode));
  }
}

poplar::Tensor
resizeNearest1D(poplar::Tensor &input, int dim, const ResizeParams &params) {
  // Check float is an int.
  // https://stackoverflow.com/a/25274904
  constexpr float eps      = 0.00001f;
  const float scale        = params.scales.at(dim);
  const float roundedScale = std::roundf(scale);
  const bool scaleIsNonNegInt =
      std::fabs(roundedScale - scale) <= eps && roundedScale >= 0.f;

  bool coordinateTransformationModeIsHalfPixel =
      params.coordinateTransformationMode ==
      ResizeCoordinateTransformationMode::HalfPixel;

  // Use poplar::Tensor::upsample if possible, as our generalised method for
  // is extremely expensive on tensor expressions. If the scale is
  // a positive integer, and the resize mode is not floor or ceil, and the
  // coordinate transformation mode is half_pixel, it is ok to use poplars
  // upsample. Poplars upsample works equally well for both resize modes,
  // round_prefer_floor, and round_prefer_ceil. If we look at the equation used
  // to transform the index:
  //   `rounding_mode((i + 0.5) / scale - 0.5)`
  // This will only return a different result if the answer to
  //   `(i + 0.5) / scale - 0.5`
  // is `x.5`. If we then take the equation:
  //   `(i + 0.5) / scale - 0.5 = 0.5`
  // and rearrange it for s, we get:
  //   `s = i + 0.5`
  // but we know both s and i have to be integers and this can not be satisfied.
  if (scaleIsNonNegInt &&
      isNotOneOf(params.nearestMode,
                 {ResizeNearestMode::Floor, ResizeNearestMode::Ceil}) &&
      coordinateTransformationModeIsHalfPixel) {
    return input.upsample(scale, dim, poplar::UpsampleMethod::REPEAT);
  } else {
    auto slices = split(input, dim);

    std::vector<poplar::Tensor> toConcat;
    for (int i = 0; i < params.outShape.at(dim); i++) {
      int idx =
          applyNearestMode(coordinateTransformation(i, dim, params), params);
      idx = clamp(idx, 0, slices.size() - 1);
      toConcat.push_back(slices.at(idx));
    }

    return poplar::concat(toConcat, dim);
  }
}

poplar::Tensor resizeNearest(poplar::Tensor input,
                             const ResizeParams &params,
                             poplar::program::Sequence &prog,
                             snap::Graph &graph,
                             poplar::DebugContext debugContext) {
  auto result = graph.getPoplarGraph().clone(input, debugContext);
  prog.add(poplar::program::Copy(input, result, false, debugContext));

  for (int i = 0; i < input.rank(); i++) {
    if (params.nearestMode == ResizeNearestMode::Pytorch) {
      if (result.shape().at(i) != params.outShape.at(i)) {
        result = resizeNearest1D(result, i, params);
      }
    } else {
      // Even if the output shape is the same, resize can still have an affect
      // on the values. Instead scale is checked.
      auto scale = params.scales.at(i);
      if (scale != 1.0f) {
        result = resizeNearest1D(result, i, params);
      }
    }
  }

  return result;
}

poplar::Tensor resizeLinear(poplar::Tensor input,
                            const ResizeParams &params,
                            poplar::program::Sequence &prog,
                            snap::Graph &graph,
                            poplar::DebugContext debugContext) {
  auto result = graph.getPoplarGraph().clone(input, debugContext);
  prog.add(poplar::program::Copy(input, result, false, debugContext));

  // Generate new params for calculating nearest floor.
  ResizeParams paramsFloor = params;
  paramsFloor.mode         = ResizeMode::Nearest;
  paramsFloor.nearestMode  = ResizeNearestMode::Floor;

  // Generate new params for calculating nearest ceiling.
  ResizeParams paramsCeil = params;
  paramsCeil.mode         = ResizeMode::Nearest;
  paramsCeil.nearestMode  = ResizeNearestMode::Ceil;

  for (int dim = 0; dim < input.rank(); dim++) {
    // Even if the output shape is the same, resize can still have an affect
    // on the values. Instead scale is checked.
    auto scale = params.scales.at(dim);
    if (scale != 1.0f) {
      auto resultFloor = resizeNearest1D(result, dim, paramsFloor);
      auto resultCeil  = resizeNearest1D(result, dim, paramsCeil);

      std::vector<float> coeffs;
      for (int outIndex = 0; outIndex < params.outShape.at(dim); outIndex++) {
        float x = coordinateTransformation(outIndex, dim, params);
        x       = std::max(x, 0.0f);
        x       = x - floor(x);
        coeffs.push_back(x);
      }
      logging::debug("Coeffs: {}", coeffs);

      std::vector<size_t> coeffsShape(resultFloor.rank(), 1);
      coeffsShape.at(dim) = coeffs.size();

      auto coeffsTensor = graph.getPoplarGraph().addConstant<float>(
          poplar::FLOAT, coeffsShape, coeffs, debugContext);
      graph.getPoplarGraph().setTileMapping(coeffsTensor, 0);
      auto oneTensor = graph.getPoplarGraph().addConstant<float>(
          poplar::FLOAT, {1}, 1.0f, debugContext);
      graph.getPoplarGraph().setTileMapping(oneTensor, 0);
      auto oneMinusCoeffs = popops::sub(
          graph.getPoplarGraph(), oneTensor, coeffsTensor, prog, debugContext);

      resultCeil = popops::mul(
          graph.getPoplarGraph(), resultCeil, coeffsTensor, prog, debugContext);
      resultFloor = popops::mul(graph.getPoplarGraph(),
                                resultFloor,
                                oneMinusCoeffs,
                                prog,
                                debugContext);
      result      = popops::add(
          graph.getPoplarGraph(), resultFloor, resultCeil, prog, debugContext);
    }
  }

  return result;
}

} // namespace

ResizeOpx::ResizeOpx(Op *op, Devicex *devicex) : PopOpx(op, devicex) {
  verifyOp<ResizeOp>(op);
}

void ResizeOpx::grow(poplar::program::Sequence &prog) const {
  auto &resizeOp = getOp<ResizeOp>();
  auto params    = getResizeParams(resizeOp);
  auto outShape  = resizeOp.outShape(ResizeOp::getOutIndex());

  auto input = getInTensor(ResizeOp::getInIndex()).getPoplarTensor();
  poplar::Tensor result;
  switch (params.mode) {
  case ResizeMode::Nearest:
    result = resizeNearest(input, params, prog, graph(), debugContext());
    break;
  case ResizeMode::Linear:
    result = resizeLinear(input, params, prog, graph(), debugContext());
    break;
  default:
    throw error("Unsupported resize mode {}", params.mode);
  }

  setOutTensor(ResizeOp::getOutIndex(), snap::Tensor{result, graph()});
}

ResizeGradOpx::ResizeGradOpx(Op *op, Devicex *devicex) : PopOpx(op, devicex) {
  verifyOp<ResizeGradOp>(op);
}

void ResizeGradOpx::grow(poplar::program::Sequence &prog) const {
  auto &op   = getOp<ResizeGradOp>();
  auto input = getInTensor(ResizeGradOp::getInIndex());

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

  setOutTensor(ResizeGradOp::getOutIndex(), result);
}

snap::Tensor ResizeGradOpx::reduceDimension(poplar::program::Sequence &prog,
                                            const snap::Tensor &input,
                                            int dimension,
                                            float scale) const {
  auto slices = split(input.getPoplarTensor(), dimension);

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
  return snap::Tensor{poplar::concat(toConcat, dimension), graph()};
}

snap::Tensor ResizeGradOpx::padDimension(poplar::program::Sequence &prog,
                                         const snap::Tensor &input,
                                         int dimension,
                                         int64_t newSize,
                                         float scale) const {
  auto slices        = split(input.getPoplarTensor(), dimension);
  auto paddingTensor = graph().getPoplarGraph().addVariable(
      input.getPoplarTensor().elementType(), slices.at(0).shape());
  popops::zero(graph().getPoplarGraph(),
               paddingTensor,
               prog,
               debugContext("zeroPadding"));

  std::vector<poplar::Tensor> toConcat(newSize, paddingTensor);
  for (int i = 0; i < slices.size(); i++) {
    int idx          = static_cast<int>(std::floor(i * scale));
    toConcat.at(idx) = slices.at(i);
  }

  return snap::Tensor{poplar::concat(toConcat, dimension), graph()};
}

namespace {
OpxCreator<ResizeOpx> resizeOpxCreator(Onnx::CustomOperators::Resize);
OpxCreator<ResizeGradOpx> resizeGradOpxCreator(Onnx::GradOperators::ResizeGrad);
} // namespace

} // namespace popx
} // namespace popart
