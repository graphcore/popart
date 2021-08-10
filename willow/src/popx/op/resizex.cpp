#include <cmath>

#include <poplin/MatMul.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Pad.hpp>
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

float coordinateTransformation(float idx,
                               ResizeCoordinateTransformationMode mode,
                               float scale,
                               ResizeNearestMode nearestMode,
                               int inputSize) {
  if (nearestMode == ResizeNearestMode::Pytorch) {
    return idx / scale;
  } else {

    switch (mode) {
    case ResizeCoordinateTransformationMode::HalfPixel:
      return (idx + 0.5f) / scale - 0.5f;
    case ResizeCoordinateTransformationMode::PytorchHalfPixel: {
      float size = static_cast<float>(inputSize) * scale;
      if (size > 1.0f) {
        return (idx + 0.5f) / scale - 0.5f;
      } else {
        return 0.0f;
      }
    }
    case ResizeCoordinateTransformationMode::Asymmetric:
      return idx / scale;
    case ResizeCoordinateTransformationMode::AlignCorners: {
      float size = static_cast<float>(inputSize) * scale;
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

// When using linear or cubic, we shouldn't need to specify nearest mode.
float coordinateTransformation(float idx,
                               ResizeCoordinateTransformationMode mode,
                               float scale,
                               int inputSize) {
  return coordinateTransformation(
      idx, mode, scale, ResizeNearestMode::N, inputSize);
}

float coordinateTransformation(float idx, int dim, const ResizeParams &params) {
  return coordinateTransformation(idx,
                                  params.coordinateTransformationMode,
                                  params.scales.at(dim),
                                  params.nearestMode,
                                  params.inShape.at(dim));
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

class ResizeCubicHelper {
public:
  poplar::Tensor input;
  snap::Graph &graph;
  const ResizeParams &params;
  poplar::program::Sequence &prog;
  poplar::DebugContext debugContext;

  // These are caches to the function `cubicCoeffs` and `interpolateNDCache`.
  std::map<std::pair<float, float>, poplar::Tensor> cubicCoeffsCache;
  std::map<std::tuple<std::vector<float>,
                      std::vector<int64_t>,
                      std::vector<int64_t>>,
           poplar::Tensor>
      interpolateNDCache;

  ResizeCubicHelper(poplar::Tensor input_,
                    snap::Graph &graph_,
                    const ResizeParams &params_,
                    poplar::program::Sequence &prog_,
                    poplar::DebugContext debugContext_)
      : input(input_), graph(graph_), params(params_), prog(prog_),
        debugContext(debugContext_) {}

  poplar::Tensor cubicCoeffs(float ratio, float A) {
    // Check if there is a cached result for this call to cubicCoeffs.
    auto found = cubicCoeffsCache.find(std::make_pair(ratio, A));
    if (found != cubicCoeffsCache.end()) {
      return found->second;
    } else {
      // See https://ieeexplore.ieee.org/document/1163711 for details.
      const std::vector<float> coeffs = {
          ((A * (ratio + 1) - 5 * A) * (ratio + 1) + 8 * A) * (ratio + 1) -
              4 * A,
          ((A + 2) * ratio - (A + 3)) * ratio * ratio + 1,
          ((A + 2) * (1 - ratio) - (A + 3)) * (1 - ratio) * (1 - ratio) + 1,
          ((A * ((1 - ratio) + 1) - 5 * A) * ((1 - ratio) + 1) + 8 * A) *
                  ((1 - ratio) + 1) -
              4 * A};
      auto coeffsTensor = graph.getPoplarGraph().addConstant<float>(
          poplar::FLOAT, {4}, coeffs, debugContext);
      graph.getPoplarGraph().setTileMapping(coeffsTensor, 0);
      cubicCoeffsCache.insert({{ratio, A}, coeffsTensor});
      return coeffsTensor;
    }
  }

  poplar::Tensor dotProd(poplar::Tensor lhs, poplar::Tensor rhs) {
    assert(lhs.dim(0) == rhs.dim(0));
    lhs = lhs.reshape({1, lhs.dim(0)});
    rhs = rhs.reshape({rhs.dim(0), 1});
    auto r =
        poplin::matMul(graph.getPoplarGraph(), lhs, rhs, prog, debugContext);
    return r.reshape({1});
  }

  poplar::Tensor interpolate1D(poplar::Tensor data, float scaleFactor, int x) {
    float x_ori = coordinateTransformation(
        x, params.coordinateTransformationMode, scaleFactor, data.dim(0));
    int x_ori_int = std::floor(x_ori);

    float ratio = x_ori - x_ori_int;

    auto coeffsTensor = cubicCoeffs(ratio, -0.75f);
    auto n            = coeffsTensor.dim(0);
    auto padded =
        popops::pad(data, n / 2, n / 2, 0, popops::padding::Type::EDGE);
    int sliceStart = std::floor(x_ori + n / 2) - 1;
    int sliceEnd   = sliceStart + coeffsTensor.dim(0);
    auto d         = padded.slice(sliceStart, sliceEnd, 0);

    return dotProd(coeffsTensor, d);
  }

  poplar::Tensor interpolateND(const std::vector<float> &scales,
                               const std::vector<int64_t> &xs,
                               const std::vector<int64_t> &slices) {
    // Check if there is a cached result for this call to cubicCoeffs.
    auto found = interpolateNDCache.find(std::make_tuple(scales, xs, slices));
    if (found != interpolateNDCache.end()) {
      return found->second;
    } else {
      poplar::Tensor result;
      if (xs.size() == 1) {
        auto d = input;
        for (auto slice : slices) {
          d = d.slice(slice, slice + 1, 0).squeeze({0});
        }

        result = interpolate1D(d, scales.at(0), xs.at(0));
      } else {
        std::vector<poplar::Tensor> elems;
        for (int i = 0; i < input.dim(slices.size()); i++) {
          auto subslices = slices;
          subslices.push_back(i);

          auto r = interpolateND(
              std::vector<float>(scales.begin() + 1, scales.end()),
              std::vector<int64_t>(xs.begin() + 1, xs.end()),
              subslices);
          elems.push_back(r);
        }
        auto r = poplar::concat(elems, 0);
        result = interpolate1D(r, scales.at(0), xs.at(0));
      }

      interpolateNDCache.insert({{scales, xs, slices}, result});
      return result;
    }
  }

  poplar::Tensor run(std::vector<int64_t> indices = {}) {
    if (indices.size() < input.rank()) {
      indices.push_back(0);

      std::vector<poplar::Tensor> elems;

      for (int i = 0; i < params.outShape.at(indices.size() - 1); i++) {
        indices.at(indices.size() - 1) = i;
        auto x                         = run(indices);
        if (indices.size() != params.outShape.size()) {
          std::vector<size_t> expansion(1, 0);
          x = x.expand(expansion);
        }

        elems.push_back(x);
      }
      return poplar::concat(elems, 0);
    } else {
      return interpolateND(params.scales, indices, {});
    }
  }
};

poplar::Tensor resizeCubic(poplar::Tensor input,
                           const ResizeParams &params,
                           poplar::program::Sequence &prog,
                           snap::Graph &graph,
                           poplar::DebugContext debugContext) {
  return ResizeCubicHelper(input, graph, params, prog, debugContext).run();
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
  case ResizeMode::Cubic:
    result = resizeCubic(input, params, prog, graph(), debugContext());
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
      input.elementType(),
      slices.at(0).shape(),
      poplar::VariableMappingMethod::LINEAR);
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
