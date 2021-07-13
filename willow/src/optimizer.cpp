// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/loss.hpp>
#include <popart/op/sgd0varupdate.hpp>
#include <popart/op/sgd1combo.hpp>
#include <popart/op/sgd2combo.hpp>
#include <popart/optimizer.hpp>
#include <popart/sessionoptions.hpp>
#include <popart/tensor.hpp>
#include <popart/tensordata.hpp>
#include <popart/tensornames.hpp>
#include <popart/tensors.hpp>
#include <popart/util.hpp>

#include <boost/functional/hash.hpp>

namespace popart {

std::map<std::string, OptimizerValue>
getOptMap(const std::map<std::string, std::pair<float, bool>> &m) {
  std::map<std::string, OptimizerValue> mOptVals;
  for (auto x : m) {
    mOptVals.insert({x.first, OptimizerValue(x.second)});
  }
  return mOptVals;
}

void Optimizer::validReplacement(const Optimizer &other) const {
  logging::ir::debug("Checking clip norm settings.");
  if (clipNormSettings.size() != other.clipNormSettings.size()) {
    throw optimizer_replacement_error("Clip norm settings do not match.");
  }

  for (int i = 0; i < clipNormSettings.size(); i++) {
    if (clipNormSettings[i] != other.clipNormSettings[i]) {
      throw optimizer_replacement_error(
          "Clip norm settings at index {} do not match.", i);
    }
  }

  logging::ir::debug("Checking optimizer types.");
  if (other.type() != type()) {
    throw optimizer_replacement_error(
        "Can not replace optimizer of type {} with new optimizer of type {}",
        type_s(),
        other.type_s());
  }
}

void Optimizer::setFactorsFromOptions(const SessionOptions &opts) {
  replicatedGraphCount       = opts.getGlobalReplicationFactor();
  enableGradientAccumulation = opts.enableGradientAccumulation;
  accumulationFactor = enableGradientAccumulation ? opts.accumulationFactor : 1;
  meanReduction =
      opts.accumulationAndReplicationReductionType == ReductionType::Mean;

  postMeanAccumulation =
      enableGradientAccumulation && meanReduction &&
      (opts.meanAccumulationAndReplicationReductionStrategy ==
           MeanReductionStrategy::Post ||
       opts.meanAccumulationAndReplicationReductionStrategy ==
           MeanReductionStrategy::PostAndLoss);

  postMeanReplication = replicatedGraphCount > 1 && meanReduction &&
                        opts.meanAccumulationAndReplicationReductionStrategy ==
                            MeanReductionStrategy::Post;
  // TODO(T42812): Remove after deprecation period
  lossMeanReplication = replicatedGraphCount > 1 && meanReduction &&
                        opts.meanAccumulationAndReplicationReductionStrategy ==
                            MeanReductionStrategy::PostAndLoss;
  factorsAreSetFromOptions = true;
}

bool Optimizer::gradientAccumulationEnabled() const {
  if (!factorsAreSetFromOptions) {
    throw error("Cannot call Optimizer::gradientAccumulationEnabled until "
                "Optimizer::setFactorsFromOptions has been called");
  }
  return enableGradientAccumulation;
}

bool Optimizer::meanReductionEnabled() const {
  if (!factorsAreSetFromOptions) {
    throw error("Cannot call Optimizer::meanReductionEnabled until "
                "Optimizer::setFactorsFromOptions has been called");
  }
  return meanReduction;
}

bool Optimizer::meanGradientAccumulationEnabled() const {
  logging::warn("Optimizer::meanGradientAccumulationEnabled is deprecated. "
                "Please use Optimizer::meanReductionEnabled instead.");
  return meanReductionEnabled();
}

bool Optimizer::postMeanAccumulationEnabled() const {
  if (!factorsAreSetFromOptions) {
    throw error("Cannot call Optimizer::postMeanAccumulationEnabled until "
                "Optimizer::setFactorsFromOptions has been called");
  }
  return postMeanAccumulation;
}

bool Optimizer::postMeanReplicationEnabled() const {
  if (!factorsAreSetFromOptions) {
    throw error("Cannot call Optimizer::postMeanReplicationEnabled until "
                "Optimizer::setFactorsFromOptions has been called");
  }
  return postMeanReplication;
}

bool Optimizer::lossMeanReplicationEnabled() const {
  if (!factorsAreSetFromOptions) {
    throw error("Cannot call Optimizer::lossMeanReplicationEnabled until "
                "Optimizer::setFactorsFromOptions has been called");
  }
  return lossMeanReplication;
}

int64_t Optimizer::getReplicatedGraphCount() const {
  if (!factorsAreSetFromOptions) {
    throw error("Cannot call Optimizer::getReplicatedGraphCount until "
                "Optimizer::setFactorsFromOptions has been called");
  }
  return replicatedGraphCount;
}

int64_t Optimizer::getAccumulationFactor() const {
  if (!factorsAreSetFromOptions) {
    throw error("Cannot call Optimizer::getAccumulationFactor until "
                "Optimizer::setFactorsFromOptions has been called");
  }
  return accumulationFactor;
}

TensorId Optimizer::getLossScalingTensorId(DataType t) {
  return reservedLossScalingPrefix() + getDataTypeInfoMap().at(t).name();
}

Optimizer::Optimizer(OptimizerValue ls_,
                     const std::vector<ClipNormSettings> &clipNormSettings_)
    : ls(ls_), clipNormSettings(clipNormSettings_) {
  // Reject loss scaling of 0.
  if (!(ls.val() > 0.0f || ls.val() < 0.0f)) {
    throw error("Loss scaling cannot be 0");
  }
}

size_t Optimizer::hash() const {
  std::size_t seed = 0;
  boost::hash_range(seed, clipNormSettings.begin(), clipNormSettings.end());
  boost::hash_combine(seed, ls);
  return seed;
}

} // namespace popart

namespace std {
std::size_t std::hash<popart::ClipNormSettings>::operator()(
    const popart::ClipNormSettings &settings) const {
  std::size_t seed = 0;
  boost::hash_combine(seed, settings.maxNorm);
  boost::hash_range(seed, settings.weightIds.begin(), settings.weightIds.end());
  return seed;
}
} // namespace std
