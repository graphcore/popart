#include <poponnx/error.hpp>
#include <poponnx/popx/convoptionsx.hpp>
#include <poponnx/popx/graphcachex.hpp>

#include <poplin/ConvUtil.hpp>
#include <poplin/Convolution.hpp>
#include <poputil/GraphFunction.hpp>

#include <iostream>
#include <map>

namespace poponnx {
namespace popx {
namespace {
bool isTrainingBwdPass(const ConvOptions &options) {
  return options.options.at("pass").compare("TRAINING_BWD") == 0;
}
} // namespace
using namespace poputil::graphfn;

GraphCachex::GraphCachex() {}

GraphCachex::PoplarTensorSignature
GraphCachex::getPoplarTensorSignature(const poplar::Tensor &tensor) {
  return {tensor.elementType(), tensor.shape()};
}

GraphCachex::ConvolutionCacheKey
GraphCachex::getConvolutionCacheKey(const poplin::ConvParams &params,
                                    const ConvOptions &options,
                                    const bool &transposeAndFlipWeights) {
  // Create signature for the convolution input
  std::vector<std::size_t> inShape = {params.getBatchSize(),
                                      params.getNumInputChans()};
  inShape.insert(inShape.end(),
                 params.inputFieldShape.begin(),
                 params.inputFieldShape.end());
  PoplarTensorSignature inSignature(params.dType, std::move(inShape));

  // Create signature for the weights
  std::vector<std::size_t> weightsShape = {
      params.getNumConvGroups(),
      params.getNumOutputChansPerConvGroup(),
      params.getNumInputChansPerConvGroup()};
  weightsShape.insert(
      weightsShape.end(), params.kernelShape.begin(), params.kernelShape.end());
  PoplarTensorSignature weightsSignature(params.dType, std::move(weightsShape));
  return std::make_tuple(inSignature,
                         weightsSignature,
                         poplin::canonicalizeParams(params),
                         options.options,
                         transposeAndFlipWeights);
}

GraphCachex::CalculateWeightDeltasCacheKey
GraphCachex::getCalculateWeightDeltasKey(const poplar::Tensor &zDeltas,
                                         const poplar::Tensor &activations,
                                         const poplin::ConvParams &params,
                                         const ConvOptions &options) {
  return std::make_tuple(getPoplarTensorSignature(zDeltas),
                         getPoplarTensorSignature(activations),
                         poplin::canonicalizeParams(params),
                         options.options);
}

GraphCachex::BwdWeightCacheKey
GraphCachex::getBwdWeightCacheKey(const poplar::Tensor &weights,
                                  const poplar::Tensor &bwdWeights) {
  return {getPoplarTensorSignature(weights),
          getPoplarTensorSignature(bwdWeights)};
}

poplar::Tensor
GraphCachex::createCachedConvolution(poplar::Graph &graph,
                                     const poplar::Tensor &in,
                                     const poplar::Tensor &weights,
                                     const poplin::ConvParams &params,
                                     bool transposeAndFlipWeights,
                                     poplar::program::Sequence &prog,
                                     bool cacheOperation,
                                     const std::string &debugPrefix,
                                     const ConvOptions &options,
                                     poplin::PlanningCache *cache) {
  std::vector<poplar::Tensor> convArgs = {in, weights};
  auto cacheKey =
      getConvolutionCacheKey(params, options, transposeAndFlipWeights);
  auto it = convolutionGraphCache.find(cacheKey);
  if (it != convolutionGraphCache.end() && cacheOperation) {
    auto &convFunction = it->second;
    return convFunction(convArgs, prog);
  }

  auto convFunction =
      TensorFunction(graph,
                     {input(in, "in"), input(weights, "weights")},
                     [&](std::vector<poplar::Tensor> &args_,
                         poplar::program::Sequence &prog_) {
                       return poplin::convolution(graph,
                                                  args_[0],
                                                  args_[1],
                                                  params,
                                                  transposeAndFlipWeights,
                                                  prog_,
                                                  debugPrefix,
                                                  options.toOptionFlags(),
                                                  cache);
                     });
  if (cacheOperation) {
    convolutionGraphCache.emplace(cacheKey, convFunction);
  }
  return convFunction(convArgs, prog);
}

poplar::Tensor
GraphCachex::cachedCalculateWeightDeltas(poplar::Graph &graph,
                                         const poplar::Tensor &zDeltas,
                                         const poplar::Tensor &activations,
                                         const poplin::ConvParams &params,
                                         poplar::program::Sequence &prog,
                                         bool cacheOperation,
                                         const std::string &debugPrefix,
                                         const ConvOptions &options,
                                         poplin::PlanningCache *cache) {
  std::vector<poplar::Tensor> calculateWeightDeltasArgs = {zDeltas,
                                                           activations};
  auto cacheKey =
      getCalculateWeightDeltasKey(zDeltas, activations, params, options);
  auto it = calculateWeightDeltasGraphCache.find(cacheKey);
  if (it != calculateWeightDeltasGraphCache.end() && cacheOperation) {
    auto &calculateWeightDeltasFunction = it->second;
    return calculateWeightDeltasFunction(calculateWeightDeltasArgs, prog);
  }

  auto calculateWeightDeltasFunction = TensorFunction(
      graph,
      {input(zDeltas, "zDeltas"), input(activations, "activations")},
      [&](std::vector<poplar::Tensor> &args_,
          poplar::program::Sequence &prog_) {
        return poplin::calculateWeightDeltas(graph,
                                             args_[0],
                                             args_[1],
                                             params,
                                             prog_,
                                             debugPrefix,
                                             options.toOptionFlags(),
                                             cache);
      });
  if (cacheOperation) {
    calculateWeightDeltasGraphCache.emplace(cacheKey,
                                            calculateWeightDeltasFunction);
  }
  return calculateWeightDeltasFunction(calculateWeightDeltasArgs, prog);
}

void GraphCachex::createCachedBwdWeights(poplar::Graph &graph,
                                         const poplar::Tensor &weights,
                                         const poplar::Tensor &bwdWeights,
                                         poplar::program::Sequence &prog,
                                         const std::string &debug_prefix) {
  std::vector<poplar::Tensor> bwdWeightsArgs = {weights, bwdWeights};
  auto cacheKey = getBwdWeightCacheKey(weights, bwdWeights);
  auto it       = bwdWeightGraphCache.find(cacheKey);
  if (it != bwdWeightGraphCache.end()) {
    auto &bwdWeightsFunction = it->second;
    bwdWeightsFunction(bwdWeightsArgs, prog);
    return;
  }
  auto bwdWeightsFunction = VoidFunction(
      graph,
      {input(weights, "weights"), output(bwdWeights, "bwdWeights")},
      [&](std::vector<poplar::Tensor> &args_,
          poplar::program::Sequence &prog_) {
        poplin::weightsTransposeChansFlipXY(
            graph, args_[0], args_[1], prog_, debug_prefix);
        return prog_;
      });
  bwdWeightGraphCache.emplace(cacheKey, bwdWeightsFunction);
  bwdWeightsFunction(bwdWeightsArgs, prog);
}

poplar::Tensor GraphCachex::convolution(poplar::Graph &graph,
                                        const poplar::Tensor &in,
                                        const poplar::Tensor &weights,
                                        const poplin::ConvParams &params,
                                        const bool transposeAndFlipWeights,
                                        poplar::program::Sequence &prog,
                                        bool cacheOperation,
                                        const std::string &debugPrefix,
                                        const ConvOptions &options,
                                        poplin::PlanningCache *cache) {
  ConvOptions convOptions    = options;
  poplar::Tensor convWeights = weights;

  // If user provides 4D weights (missing 'group' dimension), add
  // an outer dimension, size 1
  poplar::Tensor weights5D = weights;
  if (weights.rank() == 4) {
    weights5D = weights.expand({0});
  }

  bool needToTransposeAndFlipWeights = transposeAndFlipWeights;
  // If we are doing a bwd pass convolution, check if we can split it into a
  // weight weightsTransposeChansFlipXY and a fwd convolution.
  if (isTrainingBwdPass(options) && transposeAndFlipWeights && cacheOperation) {

    // Change to fwd pass options
    auto fwdOptions            = options;
    fwdOptions.options["pass"] = "TRAINING_FWD";

    auto fwdCacheKey = getConvolutionCacheKey(params, fwdOptions, false);
    if (convolutionGraphCache.count(fwdCacheKey)) {
      // We found a match - adapt the weights and change the required options.
      const auto bwdWeightsName = debugPrefix + "bwdWeights";
      convWeights               = poplin::createWeights(
          graph, params, bwdWeightsName, fwdOptions.toOptionFlags(), cache);
      createCachedBwdWeights(
          graph, weights5D, convWeights, prog, bwdWeightsName);
      needToTransposeAndFlipWeights = false;
      convOptions                   = fwdOptions;
    }
  }

  return createCachedConvolution(graph,
                                 in,
                                 convWeights,
                                 params,
                                 needToTransposeAndFlipWeights,
                                 prog,
                                 cacheOperation,
                                 debugPrefix,
                                 convOptions,
                                 cache);
}

poplar::Tensor
GraphCachex::calculateWeightDeltas(poplar::Graph &graph,
                                   const poplar::Tensor &zDeltas,
                                   const poplar::Tensor &activations,
                                   const poplin::ConvParams &params,
                                   poplar::program::Sequence &prog,
                                   bool cacheOperation,
                                   const std::string &debugPrefix,
                                   const ConvOptions &options,
                                   poplin::PlanningCache *cache) {
  return cachedCalculateWeightDeltas(graph,
                                     zDeltas,
                                     activations,
                                     params,
                                     prog,
                                     cacheOperation,
                                     debugPrefix,
                                     options,
                                     cache);
}
} // namespace popx
} // namespace poponnx
