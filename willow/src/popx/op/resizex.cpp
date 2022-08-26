// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <snap/Graph.hpp>
#include <snap/Program.hpp>
#include <snap/Tensor.hpp>
#include <tuple>
#include <utility>
#include <vector>
#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>
#include <poplar/Type.hpp>
#include <poplin/MatMul.hpp>
#include <popops/DynamicSlice.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Fill.hpp>
#include <popops/Pad.hpp>
#include <popops/Zero.hpp>
#include <poputil/TileMapping.hpp>

#include <popops/Reduce.hpp>
#include <poputil/TileMapping.hpp>
#include <popart/logging.hpp>
#include <popart/op/resize.hpp>
#include <popart/popx/op/resizex.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/error.hpp"
#include "popart/graphcoreoperators.hpp"
#include "popart/names.hpp"
#include "popart/popx/debugcontextx.hpp"
#include "popart/popx/popopx.hpp"
#include "popart/util.hpp"

namespace popart {
class Op;

namespace popx {
class Devicex;

namespace {

ResizeParams getResizeParams(const ResizeOp &x) {
  return {x.inShape(ResizeOp::getInIndex()),
          x.outShape(ResizeOp::getOutIndex()),
          x.getScales(),
          x.getMode(),
          x.getNearestMode(),
          x.getCoordinateTransformationMode()};
}

ResizeParams getResizeGradParams(const ResizeGradOp &x) {
  return {x.inShape(ResizeGradOp::getInIndex()),
          x.outShape(ResizeGradOp::getOutIndex()),
          x.getFwdScales(),
          x.getMode(),
          x.getNearestMode(),
          x.getCoordinateTransformationMode()};
}

std::vector<snap::Tensor> split(const snap::Tensor &input, int dim) {
  std::vector<snap::Tensor> result;
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

snap::Tensor resizeNearest1D(snap::Tensor &input,
                             int dim,
                             const ResizeParams &params,
                             snap::Graph &graph) {
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

    std::vector<snap::Tensor> toConcat;
    for (int i = 0; i < params.outShape.at(dim); i++) {
      int idx =
          applyNearestMode(coordinateTransformation(i, dim, params), params);
      idx = clamp(idx, 0, slices.size() - 1);
      toConcat.push_back(slices.at(idx));
    }

    return snap::concat(toConcat, dim);
  }
}

snap::Tensor resizeNearest(snap::Tensor input,
                           const ResizeParams &params,
                           snap::program::Sequence &prog,
                           snap::Graph &graph,
                           poplar::DebugContext debugContext) {
  auto result = graph.clone(input, debugContext);
  prog.add(snap::program::Copy(input, result, false, debugContext));

  for (int i = 0; i < input.rank(); i++) {
    if (params.nearestMode == ResizeNearestMode::Pytorch) {
      if (result.shape().at(i) != params.outShape.at(i)) {
        result = resizeNearest1D(result, i, params, graph);
      }
    } else {
      // Even if the output shape is the same, resize can still have an affect
      // on the values. Instead scale is checked.
      auto scale = params.scales.at(i);
      if (scale != 1.0f) {
        result = resizeNearest1D(result, i, params, graph);
      }
    }
  }

  return result;
}

snap::Tensor resizeLinear(snap::Tensor input,
                          const ResizeParams &params,
                          snap::program::Sequence &prog,
                          snap::Graph &graph,
                          poplar::DebugContext debugContext) {
  auto result = graph.clone(input, debugContext);
  prog.add(snap::program::Copy(input, result, false, debugContext));

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
      auto resultFloor = resizeNearest1D(result, dim, paramsFloor, graph);
      auto resultCeil  = resizeNearest1D(result, dim, paramsCeil, graph);

      std::vector<float> coeffs;
      for (int outIndex = 0; outIndex < params.outShape.at(dim); outIndex++) {
        float x = coordinateTransformation(outIndex,
                                           params.coordinateTransformationMode,
                                           params.scales.at(dim),
                                           params.inShape.at(dim));
        x       = std::max(x, 0.0f);
        x       = x - floor(x);
        coeffs.push_back(x);
      }

      std::vector<size_t> coeffsShape(resultFloor.rank(), 1);
      coeffsShape.at(dim) = coeffs.size();

      auto coeffsTensor = graph.addConstant<float>(
          poplar::FLOAT, coeffsShape, coeffs, debugContext);
      auto oneTensor =
          graph.addConstant<float>(poplar::FLOAT, {1}, 1.0f, debugContext);
      auto oneMinusCoeffs = popops::sub(graph.getPoplarGraph(),
                                        oneTensor.getPoplarTensor(),
                                        coeffsTensor.getPoplarTensor(),
                                        prog.getPoplarSequence(),
                                        debugContext);

      resultCeil  = snap::Tensor{popops::mul(graph.getPoplarGraph(),
                                            resultCeil.getPoplarTensor(),
                                            coeffsTensor.getPoplarTensor(),
                                            prog.getPoplarSequence(),
                                            debugContext),
                                graph};
      resultFloor = snap::Tensor{popops::mul(graph.getPoplarGraph(),
                                             resultFloor.getPoplarTensor(),
                                             oneMinusCoeffs,
                                             prog.getPoplarSequence(),
                                             debugContext),
                                 graph};
      result      = snap::Tensor{popops::add(graph.getPoplarGraph(),
                                        resultFloor.getPoplarTensor(),
                                        resultCeil.getPoplarTensor(),
                                        prog.getPoplarSequence(),
                                        debugContext),
                            graph};
    }
  }

  return result;
}

class ResizeCubicHelper {
public:
  poplar::Tensor input;
  snap::Graph &graph;
  const ResizeParams &params;
  snap::program::Sequence &prog;
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
                    snap::program::Sequence &prog_,
                    poplar::DebugContext debugContext_)
      : input(input_), graph(graph_), params(params_), prog(prog_),
        debugContext(debugContext_) {}

  std::vector<float> cubicCoeffs(float ratio, float A) {
    // Check if there is a cached result for this call to cubicCoeffs.
    // See https://ieeexplore.ieee.org/document/1163711 for details.
    const std::vector<float> coeffs = {
        ((A * (ratio + 1) - 5 * A) * (ratio + 1) + 8 * A) * (ratio + 1) - 4 * A,
        ((A + 2) * ratio - (A + 3)) * ratio * ratio + 1,
        ((A + 2) * (1 - ratio) - (A + 3)) * (1 - ratio) * (1 - ratio) + 1,
        ((A * ((1 - ratio) + 1) - 5 * A) * ((1 - ratio) + 1) + 8 * A) *
                ((1 - ratio) + 1) -
            4 * A};
    return coeffs;
  }

  // will use edge pixels to pad indexes
  std::pair<std::vector<unsigned>, poplar::Tensor>
  getXOrisNCoeffsTensor(uint64_t inSize, float scaleFactor, uint64_t outSize) {
    std::vector<float> coeffs;
    std::vector<unsigned> x_oris;
    for (int64_t x = 0; x < outSize; ++x) {
      float x_ori = coordinateTransformation(
          x, params.coordinateTransformationMode, scaleFactor, inSize);
      int x_ori_int = std::floor(x_ori);
      x_oris.push_back(unsigned(x_ori_int));

      float ratio = x_ori - x_ori_int;

      auto coeffs_partial = cubicCoeffs(ratio, -0.75f);
      coeffs.insert(coeffs.end(), coeffs_partial.begin(), coeffs_partial.end());
    }
    auto coeffsTensor =
        createFloatConstant({outSize, 4},
                            poplar::DebugContext(debugContext, "coeffsTensor"),
                            coeffs);

    std::vector<unsigned> indexes;
    for (int j = 0; j < x_oris.size(); ++j) {
      // for each index we take pixels with relative offsets: -1, 0, 1, 2
      for (int i = -1; i < 3; ++i) {
        int index = x_oris[j] + i;
        // pad out of range pixels with edge value
        index = index < 0 ? 0 : index;
        index = index > inSize - 1 ? inSize - 1 : index;
        indexes.push_back(index);
      }
    }

    return {std::move(indexes), coeffsTensor};
  }

  // take the input by index
  // returned tensor shape: (otherdims..., dim, 1)
  poplar::Tensor indexTake(poplar::Tensor data,
                           std::vector<unsigned> indexes,
                           std::size_t dim) {

    data = popops::multiSlice(graph.getPoplarGraph(),
                              data,
                              indexes,
                              dim,
                              prog.getPoplarSequence(),
                              poplar::DebugContext(debugContext, "multiSlice"));

    auto result = data.dimRoll(0, data.shape().size() - 1);
    result      = result.dimRoll(dim, data.shape().size() - 1);
    return result;
  }

  // transform from input -> output indexes
  // to output->input indexes
  std::vector<std::vector<unsigned>> reverseIndex(std::vector<unsigned> indexes,
                                                  uint64_t expected_size,
                                                  uint64_t pad_index) {
    std::vector<std::vector<unsigned>> result;
    std::set<uint64_t> used_indexes;
    // if there's any index not used
    while (true) {
      std::vector<unsigned> one_seq;
      one_seq.resize(expected_size, pad_index);
      for (int64_t i = indexes.size() - 1; i >= 0; --i) {
        if (indexes[i] == -1) {
          continue;
        }
        if (used_indexes.find(indexes[i]) == used_indexes.end()) {
          one_seq[indexes[i]] = i;
          used_indexes.insert(indexes[i]);
          indexes[i] = -1;
        }
      }
      if (used_indexes.size() == 0) {
        break;
      }
      result.push_back(one_seq);
      used_indexes.clear();
    }
    return result;
  }

  poplar::Tensor createFloatConstant(std::vector<uint64_t> shape,
                                     poplar::DebugContext debugContext = "",
                                     std::vector<float> values         = {},
                                     float defaultValue                = 0) {
    uint64_t total = 1;
    for (auto i = 0; i < shape.size(); ++i) {
      total *= shape[i];
    }
    if (values.size() == 0) {
      values.resize(total, defaultValue);
    }
    auto result = graph.addConstant<float>(
        poplar::FLOAT,
        shape,
        values,
        poplar::DebugContext(debugContext, "addConstant"));
    return result.getPoplarTensor();
  }

  snap::Tensor run(std::vector<int64_t> indices = {}) {
    auto result = input;
    // do interpolation for each dimension, start from the last dimension
    for (uint64_t dim = params.scales.size() - 1; dim < params.scales.size();
         --dim) {
      auto scale = params.scales[dim];
      // if scale == 1
      // we do not need to do anything to this dim
      if (scale == 1) {
        continue;
      }

      auto x_oris_and_coeffs_tensor =
          getXOrisNCoeffsTensor(result.dim(dim), scale, params.outShape[dim]);
      std::vector<unsigned> x_oris = x_oris_and_coeffs_tensor.first;
      poplar::Tensor coeffsTensor  = x_oris_and_coeffs_tensor.second;

      result                      = indexTake(result, x_oris, std::size_t(dim));
      std::vector<uint64_t> shape = result.shape();
      // last indexes repeated for 4 times
      shape[shape.size() - 1] = 4;
      shape[shape.size() - 2] /= 4;
      // put 4 pixels at last dim
      result = result.reshape(shape);
      result = popops::mul(graph.getPoplarGraph(),
                           result,
                           coeffsTensor,
                           prog.getPoplarSequence(),
                           poplar::DebugContext(debugContext, "mulCoeffs"));
      result =
          popops::reduce(graph.getPoplarGraph(),
                         result,
                         {result.shape().size() - 1},
                         {popops::Operation::ADD},
                         prog.getPoplarSequence(),
                         poplar::DebugContext(debugContext, "reduce4Pixels"));

      // move the indexes dim back from backward to it's original position
      result = result.dimRoll(result.shape().size() - 1, dim);
    }
    return snap::Tensor{result, graph};
  }

  snap::Tensor runBackward(std::vector<int64_t> indices = {}) {
    auto result = input;
    // when running backward pass
    // we need to start from the higher rank,
    // reversed order of forward pass
    for (uint64_t dim = 0; dim < params.scales.size(); ++dim) {
      auto scale = params.scales[dim];
      // if scale == 1
      // we do not need to do anything to this dim
      if (scale == 1) {
        continue;
      }
      // we need to use params in forward pass
      auto x_oris_and_coeffs_tensor = getXOrisNCoeffsTensor(
          params.outShape.at(dim), scale, params.inShape[dim]);
      std::vector<unsigned> x_oris = x_oris_and_coeffs_tensor.first;

      poplar::Tensor coeffsTensor = x_oris_and_coeffs_tensor.second;
      // for example, current shape of result is [1,32,7,14]
      // put current dim to the end
      // [1,32,7,14] -> [1,32,14,7] for dim =2
      result = result.dimRoll(dim, result.shape().size() - 1);
      // for each dimension, we expand the each pixel to 4
      // [1,32,14,7] -> [1,32,14,7,1]
      result = result.expand({result.shape().size()});
      // [1,32,14,7,1] -> [1,32,14,7,4]
      result = poplar::concat({result, result, result, result},
                              result.shape().size() - 1);
      // we don't create new coeffTensors but use tensors in forward pass
      result = popops::mul(graph.getPoplarGraph(),
                           result,
                           coeffsTensor,
                           prog.getPoplarSequence(),
                           poplar::DebugContext(debugContext, "mulCoeffs"));
      // flatten last 2 dims
      auto shape              = result.shape();
      auto last_dim           = shape[shape.size() - 1];
      auto last2_dim          = shape[shape.size() - 2];
      shape[shape.size() - 2] = last_dim * last2_dim;
      shape.resize(shape.size() - 1);
      // [1,32,14,7,4] -> [1,32,14,28]
      result               = result.reshape(shape);
      auto reverse_indexes = reverseIndex(
          x_oris, params.outShape[dim], uint64_t(shape[shape.size() - 1]));

      // pad 0 as value for unused index
      shape[shape.size() - 1] = 1;
      auto pad                = createFloatConstant(
          shape, poplar::DebugContext(debugContext, "padZero"));
      result = poplar::concat({result, pad}, shape.size() - 1);

      // then put the value to it's original position using index
      std::vector<poplar::Tensor> elems;
      for (auto index : reverse_indexes) {
        // [1,32,14,28]->[1,32,14,14,1]
        elems.push_back(indexTake(result, index, result.shape().size() - 1));
      }
      // [1,32,14,14,1]->[1,32,14,14,n]
      // result = poplar::concat(elems, elems[0].shape().size() - 1);
      // multiple add will consume less memory than concat + reduceSum
      result = elems[0];
      for (int i = 1; i < elems.size(); ++i) {
        result =
            popops::add(graph.getPoplarGraph(),
                        elems[i],
                        result,
                        prog.getPoplarSequence(),
                        poplar::DebugContext(debugContext, "addInverseGrads"));
      }
      shape = result.shape();
      shape.resize(shape.size() - 1);
      result = result.reshape(shape);
      // move the indexes dim back from backward to it's original position
      // [1,32,14,14]->[1,32,14,14]
      result = result.dimRoll(result.shape().size() - 1, dim);
    }
    return snap::Tensor{result, graph};
  }

}; // namespace

snap::Tensor resizeCubic(snap::Tensor input,
                         const ResizeParams &params,
                         snap::program::Sequence &prog,
                         snap::Graph &graph,
                         poplar::DebugContext debugContext) {
  return ResizeCubicHelper(
             input.getPoplarTensor(), graph, params, prog, debugContext)
      .run();
}
} // namespace

ResizeOpx::ResizeOpx(Op *op, Devicex *devicex) : PopOpx(op, devicex) {
  verifyOp<ResizeOp>(op);
}

void ResizeOpx::grow(snap::program::Sequence &prog) const {
  auto &resizeOp = getOp<ResizeOp>();
  auto params    = getResizeParams(resizeOp);
  auto outShape  = resizeOp.outShape(ResizeOp::getOutIndex());

  auto input = getInTensor(ResizeOp::getInIndex());
  snap::Tensor result;
  switch (params.mode) {
  case ResizeMode::Nearest:
    result = resizeNearest(
        input, params, prog, graph(), debugContext("resizeNearest"));
    break;
  case ResizeMode::Linear:
    result = resizeLinear(
        input, params, prog, graph(), debugContext("resizeLinear"));
    break;
  case ResizeMode::Cubic:
    result =
        resizeCubic(input, params, prog, graph(), debugContext("resizeCubic"));
    break;
  default:
    throw error("Unsupported resize mode {}", params.mode);
  }

  setOutTensor(ResizeOp::getOutIndex(), result);
}

ResizeGradOpx::ResizeGradOpx(Op *op, Devicex *devicex) : PopOpx(op, devicex) {
  verifyOp<ResizeGradOp>(op);
}
snap::Tensor
ResizeGradOpx::resizeNearestGrad(ResizeGradOp &op,
                                 const snap::Tensor &input,
                                 ResizeParams &params,
                                 snap::program::Sequence &prog) const {

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
  return result;
}

void ResizeGradOpx::grow(snap::program::Sequence &prog) const {
  auto &op   = getOp<ResizeGradOp>();
  auto input = getInTensor(ResizeGradOp::getInIndex());

  auto params = getResizeGradParams(op);
  if (params.mode == ResizeMode::Nearest) {
    auto result = resizeNearestGrad(op, input, params, prog);
    setOutTensor(ResizeGradOp::getOutIndex(), result);
  } else if (params.mode == ResizeMode::Linear) {
    throw error("Resize mode 'linear' is not supported when training.");
  } else if (params.mode == ResizeMode::Cubic) {
    auto result = ResizeCubicHelper(input.getPoplarTensor(),
                                    graph(),
                                    params,
                                    prog,
                                    debugContext("reszeCubicGrad"))
                      .runBackward();
    setOutTensor(ResizeGradOp::getOutIndex(), result);
  }
}

snap::Tensor ResizeGradOpx::reduceDimension(snap::program::Sequence &prog,
                                            const snap::Tensor &input,
                                            int dimension,
                                            float scale) const {

  auto &op      = getOp<ResizeGradOp>();
  auto outShape = op.outShape(ResizeGradOp::getOutIndex());
  auto one      = graph().addConstant(
      input.elementType(), {}, 1.0f, debugContext("const_one"));

  std::vector<size_t> resultShape = input.shape();
  resultShape.at(dimension)       = outShape.at(dimension);
  auto size                       = input.getPoplarTensor().dim(dimension);
  auto result                     = graph().addVariable(
      input.elementType(), resultShape, debugContext("reduceDimResult"));
  poputil::mapTensorLinearly(graph().getPoplarGraph(),
                             result.getPoplarTensor());
  popops::fill(graph().getPoplarGraph(),
               result.getPoplarTensor(),
               prog.getPoplarSequence(),
               0.0f,
               debugContext("resultFill"));
  std::vector<unsigned int> offsets;
  offsets.reserve(size);
  for (int i = 0; i < size; i++) {
    offsets.push_back(static_cast<unsigned int>(std::floor(i * scale)));
  }
  popops::multiUpdateAdd(graph().getPoplarGraph(),
                         result.getPoplarTensor(),
                         input.getPoplarTensor()
                             .dimRoll(static_cast<unsigned>(dimension))
                             .expand({static_cast<unsigned>(dimension + 1)}),
                         offsets,
                         one.getPoplarTensor(),
                         static_cast<size_t>(dimension),
                         prog.getPoplarSequence(),
                         debugContext("reduceDimAdd"));
  return result;
}

snap::Tensor ResizeGradOpx::padDimension(snap::program::Sequence &prog,
                                         const snap::Tensor &input,
                                         int dimension,
                                         int64_t newSize,
                                         float scale) const {
  auto slices        = split(input, dimension);
  auto paddingTensor = graph().addVariable(
      input.elementType(), slices.at(0).shape(), debugContext());
  poputil::mapTensorLinearly(graph().getPoplarGraph(),
                             paddingTensor.getPoplarTensor());
  popops::zero(graph().getPoplarGraph(),
               paddingTensor.getPoplarTensor(),
               prog.getPoplarSequence(),
               debugContext("zeroPadding"));

  std::vector<snap::Tensor> toConcat(newSize, paddingTensor);
  for (int i = 0; i < slices.size(); i++) {
    int idx          = static_cast<int>(std::floor(i * scale));
    toConcat.at(idx) = slices.at(i);
  }

  return snap::concat(toConcat, dimension);
}

namespace {
OpxCreator<ResizeOpx> resizeOpxCreator(Onnx::CustomOperators::Resize);
OpxCreator<ResizeGradOpx> resizeGradOpxCreator(Onnx::GradOperators::ResizeGrad);
} // namespace

} // namespace popx
} // namespace popart
