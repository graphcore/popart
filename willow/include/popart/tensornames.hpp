// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_TENSORNAMES_HPP
#define GUARD_NEURALNET_TENSORNAMES_HPP

#include <string>
#include <vector>
#include <popart/names.hpp>

namespace popart {
class Graph;

/**
 * Given the id of a tensor \p fwdId in a graph #fwdGraph, and its corresponding
 * backward graph \p bwdGraph, returns the id of the corresponding grad tensor
 * in the \p bwdGraph.
 *
 * \param fwdGraph The forward graph.
 * \param bwdGraph A backward graph of \p fwdGraph.
 * \param fwdId The id of a tensor in \p fwdGraph.
 *
 * \returns The id of the grad tensor of \p fwdId in \p bwdGraph.
 */
TensorId fwdIdToBwdGradId(const Graph &fwdGraph,
                          const Graph &bwdGraph,
                          const TensorId &fwdId);

/**
 * Given the id of a grad tensor \p bwdId in a backward graph \p bwdGraph,
 * generated by the autodiff of graph \p fwdGraph, returns the id of the forward
 * tensor in \p fwdGraph for which this tensor is the grad of.
 *
 * \param fwdGraph The forward graph from which \p bwdGraph was generated.
 * \param bwdGraph The backward graph.
 * \param bwdId The id of a tensor in \p bwdGraph.
 *
 * \returns The id of the forward tensor of \p bwdId in \p fwdGraph.
 */
TensorId bwdGradIdToFwdId(const Graph &fwdGraph,
                          const Graph &bwdGraph,
                          const TensorId &bwdId);

/**
 * Given the id of a tensor \p fwdId in a graph \p fwdGraph, and its
 * corresponding backward graph \p bwdGraph, returns the id of the equivalent
 * cloned tensor in the \p bwdGraph. For example, if a forward tensor is
 * required in the calculation of a grad op in the backward graph, this function
 * will give the id of the clone of that tensor in the backward graph.
 *
 * \param fwdGraph The forward graph.
 * \param bwdGraph A backward graph of \p fwdGraph.
 * \param fwdId The id of a tensor in \p fwdGraph.
 *
 * \returns The id of the cloned tensor of \p fwdId in \p bwdGraph.
 */
TensorId fwdIdToClonedBwdId(const Graph &fwdGraph,
                            const Graph &bwdGraph,
                            const TensorId &fwdId);

/**
 * Given the id of a tensor \p bwdId in a graph \p bwdGraph, generated by
 * autodiff of the graph \p fwdGraph, where \p bwdId is a clone of a tensor in
 * the \p fwdGraph, returns the id of the original tensor in the \p fwdGraph.
 *
 * \param fwdGraph The forward graph from which \p bwdGraph was generated.
 * \param bwdGraph The backward graph.
 * \param bwdId The id of a tensor in \p bwdGraph.
 *
 * \returns The id of the original tensor in \p fwdGraph that was cloned to make
 *   \p bwdId.
 */
TensorId bwdNonGradIdToFwdId(const Graph &fwdGraph,
                             const Graph &bwdGraph,
                             const TensorId &bwdId);

/**
 * Creates a TensorId name which is the total gradient of a forward tensor
 * \param tenId The id to construct the name of the forward tensor total
 * gradient from \return A new TensorId with forward tensor total gradient
 * prefix
 */
TensorId getGradId(const TensorId &tenId);

/**
 * Creates a TensorId where the gradient prefix is removed
 * \param tenId The TensorId which the return value will be based upon
 * \return The new TensorId, where the gradient prefix is removed
 */
TensorId getNonGradId(const TensorId &tenId);

/**
 * Checks whether the TensorId has the reserved gradient prefix
 * \return True if the TensorId has the reserved gradient prefix
 */
bool isGradId(const TensorId &id);

/**
 * Creates a TensorId of the edge-gradient along the edge where tensor is
 * consumed by \a opId at index \a index
 *
 * Note: We don't need the name of the tensor which the resulting TensorId is an
 * edge-grad of, the edge-gradient is uniquely defined by the the edge it flows
 * on in the forward pass (input at 'index' to 'opId')
 *
 * \param opId The id of the operator consuming tenId
 * \param index The index at which the operator is consuming the tensor
 * \return The new TensorId of the edge-gradient
 */
TensorId getEdgeGradId(const OpId &opId, const int &index);

// Functions using reserved prefixes

/**
 * Creates a TensorId with the remote arg prefix
 * \param base_id The id to create the new TensorId from
 * \return A new TensorId with the remote arg prefix
 */
TensorId getRemoteArgTensorId(const TensorId &base_id);

/**
 * Creates a TensorId with the post-update prefix
 * \param id The id to create the new TensorId from
 * \return A new TensorId with the post-update prefix
 */
TensorId getUpdatedVarId(const TensorId &id);

// Functions creating reserved suffixes

/**
 * Creates a TensorId with a __re suffix
 * \param base_id The id to create the new TensorId from
 * \return A new TensorId with the __re suffix
 */
TensorId createRecomputedTensorId(const TensorId &base_id);

// Removing of prefixes

/**
 * Creates a new TensorId where all prefixes are removed
 * \param id The id to make the stripped TensorId from
 * \return A new TensorId where the prefixes are stripped away
 */
TensorId stripAllReservedPrefixes(const TensorId &id);

// Collection of prefixes

/**
 * Returns a vector containing all optimizer prefixes. These
 * include prefixes for optimizer parameters such as learning
 * rate, momentum factor, and weight decay factor. These need
 * to be identifiable when copying optimizer settings to/from
 * the IPU
 * \return A vector containing all optimizer prefixes
 */
const std::vector<std::string> &reservedOptimizerPrefixes();

/**
 * Returns a vector containing all optimizer state prefixes.
 * These include prefixes for variables that constitute the
 * optimizer state such as accumulation and step tensors
 * \return A vector containing all optimizer state prefixes
 */
const std::vector<std::string> &reservedOptimizerStatePrefixes();

/**
 * Returns a vector containing all accumulator prefixes
 * \return A vector containing all accumulator prefixes
 */
const std::vector<std::string> &reservedAccumulatorPrefixes();

/**
 * Returns a vector containing all accumulator prefixes
 * \return A vector containing all accumulator prefixes
 */
const std::vector<std::string> &reservedAccumulatorPrefixes();

/**
 * Returns a vector containing all uncategorized prefixes
 * An uncategorized prefix is one which is not included in neither
 * reservedOptimizerPrefixes, reservedOptimizerStatePrefixes,
 * reservedAccumulatorPrefixes nor reservedAccumulatorPrefixes \return A vector
 * containing all uncategorized prefixes
 */
const std::vector<std::string> &uncategorizedReservedPrefixes();

/**
 * Returns a vector containing all reserved prefixes
 * \return A vector containing all reserved prefixes
 */
const std::vector<std::string> &reservedPrefixes();

// Prefixes

// reservedOptimizerPrefixes
// Optimizer (part of reservedOptimizerPrefixes)
constexpr const char *reservedLossScalingPrefix() { return "lossScaling_"; }
constexpr const char *reservedAutomaticLossScalePrefix() {
  return "AutomaticLossScaleProxy___";
}
// SGD0 / SGD1 / SGD2 (part of reservedOptimizerPrefixes)
constexpr const char *reservedDefaultScaledLearningRate0Prefix() {
  return "scaledLearningRate0___default___";
}
constexpr const char *reservedSpecificScaledLearningRate0Prefix() {
  return "scaledLearningRate0___specific___";
}
constexpr const char *reservedDefaultWeightDecayScaleFactor0Prefix() {
  return "weightDecayScaleFactor0___default___";
}
constexpr const char *reservedSpecificWeightDecayScaleFactor0Prefix() {
  return "weightDecayScaleFactor0___specific___";
}
constexpr const char *reservedDefaultScaledLearningRate1Prefix() {
  return "scaledLearningRate1___default___";
}
constexpr const char *reservedSpecificScaledLearningRate1Prefix() {
  return "scaledLearningRate1___specific___";
}
constexpr const char *reservedDefaultScaledWeightDecay1Prefix() {
  return "scaledWeightDecay1___default___";
}
constexpr const char *reservedSpecificScaledWeightDecay1Prefix() {
  return "scaledWeightDecay1___specific___";
}
constexpr const char *reservedDefaultDampeningScaleFactor1Prefix() {
  return "dampeningScaleFactor1___default___";
}
constexpr const char *reservedSpecificDampeningScaleFactor1Prefix() {
  return "dampeningScaleFactor1___specific___";
}
constexpr const char *reservedDefaultScaledMomentum1Prefix() {
  return "scaledMomentum1___default___";
}
constexpr const char *reservedSpecificScaledMomentum1Prefix() {
  return "scaledMomentum1___specific___";
}
constexpr const char *reservedDefaultScaledLearningRate2Prefix() {
  return "scaledLearningRate2___default___";
}
constexpr const char *reservedSpecificScaledLearningRate2Prefix() {
  return "scaledLearningRate2___specific___";
}
constexpr const char *reservedDefaultDampeningScaleFactor2Prefix() {
  return "dampeningScaleFactor2___default___";
}
constexpr const char *reservedSpecificDampeningScaleFactor2Prefix() {
  return "dampeningScaleFactor2___specific___";
}
constexpr const char *reservedDefaultScaledMomentum2Prefix() {
  return "scaledMomentum2___default___";
}
constexpr const char *reservedSpecificScaledMomentum2Prefix() {
  return "scaledMomentum2___specific___";
}
// Adam / Lamb (part of reservedOptimizerPrefixes)
constexpr const char *reservedDefaultAdamBeta1Prefix() {
  return "adamBeta1___default___";
}
constexpr const char *reservedSpecificAdamBeta1Prefix() {
  return "adamBeta1___specific___";
}
constexpr const char *reservedDefaultAdamBeta2Prefix() {
  return "adamBeta2___default___";
}
constexpr const char *reservedSpecificAdamBeta2Prefix() {
  return "adamBeta2___specific___";
}
constexpr const char *reservedDefaultLearningRatePrefix() {
  return "learningRate___default___";
}
constexpr const char *reservedSpecificLearningRatePrefix() {
  return "learningRate___specific___";
}
constexpr const char *reservedDefaultAdamEpsPrefix() {
  return "adamEps___default___";
}
constexpr const char *reservedSpecificAdamEpsPrefix() {
  return "adamEps___specific___";
}
constexpr const char *reservedDefaultWeightDecayPrefix() {
  return "weightDecay___default___";
}
constexpr const char *reservedSpecificWeightDecayPrefix() {
  return "weightDecay___specific___";
}
constexpr const char *reservedDefaultAdamGradientScalingPrefix() {
  return "adamGradientScaling___default___";
}
constexpr const char *reservedSpecificAdamGradientScalingPrefix() {
  return "adamGradientScaling___specific___";
}
constexpr const char *reservedDefaultMaxWeightNormPrefix() {
  return "maxWeightNorm___default___";
}
constexpr const char *reservedSpecificMaxWeightNormPrefix() {
  return "maxWeightNorm___specific___";
}
// Adaptive (part of reservedOptimizerPrefixes)
constexpr const char *reservedDefaultAdaptiveAlphaPrefix() {
  return "adaptiveAlpha___default___";
}
constexpr const char *reservedSpecificAdaptiveAlphaPrefix() {
  return "adaptiveAlpha___specific___";
}
constexpr const char *reservedDefaultAdaptiveEpsPrefix() {
  return "adaptiveEps___default___";
}
constexpr const char *reservedSpecificAdaptiveEpsPrefix() {
  return "adaptiveEps___specific___";
}
constexpr const char *reservedDefaultAdaptiveGradientScalingPrefix() {
  return "adaptiveGradientScaling___default___";
}
constexpr const char *reservedSpecificAdaptiveGradientScalingPrefix() {
  return "adaptiveGradientScaling___specific___";
}
constexpr const char *reservedDefaultAdaptiveMomentumPrefix() {
  return "adaptiveMomentum___default___";
}
constexpr const char *reservedSpecificAdaptiveMomentumPrefix() {
  return "adaptiveMomentum___specific___";
}

// reservedAccumulatorPrefixes
constexpr const char *reservedAccumPrefix() { return "Accum___"; }
// Accumulator/momentum combination tensors, specifically used for the SGD1
// optimizer
constexpr const char *reservedAcclPrefix() { return "Accl___"; }
constexpr const char *reservedAcclToReducePrefix() {
  return "AcclIntoReduce___";
}
constexpr const char *reservedAcclToUpdatePrefix() {
  return "AcclIntoUpdate___";
}
constexpr const char *reservedAcclFinalOutPrefix() {
  return "AcclOutOfAcclUpdate___";
}
constexpr const char *reservedCounterPrefix() { return "Counter___"; }

// Ucategorized (alphabethically sorted)

constexpr const char *reservedAccl1Prefix() { return "Accl1___"; }
constexpr const char *reservedAccl2Prefix() { return "Accl2___"; }
constexpr const char *reservedAccl3Prefix() { return "Accl3___"; }
constexpr const char *reservedAdamUpdaterPrefix() { return "AdamUpdater___"; }
constexpr const char *reservedAdaptiveUpdaterPrefix() {
  return "AdaptiveUpdater___";
}
constexpr const char *anchorSumPrefix() { return "anchorSum___"; }
constexpr const char *anchorFinalPrefix() { return "anchorFinal___"; }
constexpr const char *reservedConcatInitPrefix() { return "ConcatInit___"; }
constexpr const char *reservedConstValuePrefix() { return "ConstValue___"; }
constexpr const char *cycleCountPrefix() { return "cycleCount___"; }
constexpr const char *reservedFinalReducedGradPrefix() {
  return "FinalReducedGradient___";
}
constexpr const char *reservedGlobalNormPrefix() { return "GlobalNorm___"; }
constexpr const char *reservedGradientPrefix() { return "Gradient___"; }
constexpr const char *reservedIndexPrefix() { return "Index___"; }
constexpr const char *reservedInitPrefix() { return "Init___"; }
constexpr const char *reservedLoopIteratorPrefix() { return "Iterator___"; }
constexpr const char *reservedLambR1SqPrefix() { return "LambR1Sq___"; }
constexpr const char *reservedLambR2SqPrefix() { return "LambR2Sq___"; }
constexpr const char *reservedLoopCondPrefix() { return "LoopCond___"; }
constexpr const char *reservedDefaultLossScalingPrefix() {
  return "lossScaling___default___";
}
constexpr const char *reservedSpecificLossScalingPrefix() {
  return "lossScaling___specific___";
}
constexpr const char *reservedLossScalingRatioPrefix() {
  return "LossScalingRatio_";
}
constexpr const char *reservedPreviousLossScalingPrefix() {
  return "PreviousLossScaling_";
}
constexpr const char *reservedRandomSeedPrefix() { return "randomSeed___"; }
constexpr const char *reservedSeedModifierPrefix() {
  return "randomSeedModifier___";
}
constexpr const char *reservedRemoteArgPrefix() { return "RemoteArg___"; }
constexpr const char *reservedRestoredPrefix() { return "Restored___"; }
constexpr const char *reservedStashedPrefix() { return "Stashed___"; }
constexpr const char *reservedStepPrefix() { return "Step___"; }
constexpr const char *reservedDefaultStepPrefix() {
  return "step___default___";
}
constexpr const char *reservedSpecificStepPrefix() {
  return "step___specific___";
}
constexpr const char *reservedUpdatedVarPrefix() { return "UpdatedVar___"; }

} // namespace popart

#endif
