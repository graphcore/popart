// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <boost/functional/hash.hpp>
#include <cstddef>
#include <cstdint>
#include <map>
#include <ostream>
#include <string>
#include <utility>
#include <vector>
#include <popart/error.hpp>
#include <popart/optimizer.hpp>
#include <popart/sessionoptions.hpp>
#include <popart/tensornames.hpp>

#include "popart/clipnormsettings.hpp"
#include "popart/datatype.hpp"
#include "popart/debugcontext.hpp"
#include "popart/logging.hpp"
#include "popart/op.hpp"
#include "popart/optimizervalue.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/tensorinfo.hpp"

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

  postMeanAccumulation = enableGradientAccumulation && meanReduction &&
                         opts.meanAccumulationAndReplicationReductionStrategy ==
                             MeanReductionStrategy::Post;

  postMeanReplication = replicatedGraphCount > 1 && meanReduction &&
                        opts.meanAccumulationAndReplicationReductionStrategy ==
                            MeanReductionStrategy::Post;
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

float Optimizer::getFinalLossScalingVal() const {
  if (!factorsAreSetFromOptions) {
    throw error("Cannot call Optimizer::getLossScalingVal until "
                "Optimizer::setFactorsFromOptions has been called");
  }

  float lossScalingVal = ls.val();

  return lossScalingVal;
}

TensorId Optimizer::getLossScalingTensorId(DataType t) {
  return reservedLossScalingPrefix() + getDataTypeInfoMap().at(t).name();
}

Optimizer::Optimizer(OptimizerValue ls_,
                     const std::vector<ClipNormSettings> &clipNormSettings_,
                     const DebugContext &debugContext_)
    : debugContext(debugContext_), ls(ls_),
      clipNormSettings(clipNormSettings_) {
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

std::ostream &operator<<(std::ostream &os, const OptimizerReductionType &ort) {
  switch (ort) {
  case OptimizerReductionType::None:
    os << "None";
    break;
  case OptimizerReductionType::GradReduce:
    os << "GradReduce";
    break;
  case OptimizerReductionType::AcclReduce:
    os << "AcclReduce";
    break;
  case OptimizerReductionType::AccumReduce:
    os << "AccumReduce";
    break;
  default: {
    throw error("Unexpected value for OptimizerReductionType {}",
                static_cast<int>(ort));
  }
  }
  return os;
}

std::ostream &operator<<(std::ostream &os, const WeightDecayMode &wdm) {
  switch (wdm) {
  case WeightDecayMode::Decay:
    os << "Decay";
    break;
  case WeightDecayMode::L2Regularization:
    os << "L2Regularization";
    break;
  default: {
    throw error("Unexpected value for WeightDecayMode {}",
                static_cast<int>(wdm));
  }
  }
  return os;
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
