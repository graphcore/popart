// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <popart/clipnormsettings.hpp>
#include <popart/error.hpp>
#include <popart/logging.hpp>
#include <popart/util.hpp>

namespace popart {

ClipNormSettings::ClipNormSettings(const std::vector<TensorId> &weightIds_,
                                   float maxNorm_)
    : weightIds(weightIds_), maxNorm(maxNorm_),
      mode(Mode::ClipSpecifiedWeights) {
  logging::warn(
      "Using deprecated constructor ClipNormSettings(const "
      "std::vector<TensorId> &weightIds_, float maxNorm_). The members "
      "'weightIds' and 'maxNorm' have also been deprecated. These shall all be "
      "removed from a future release. Please use the static creation methods, "
      "'clipWeights' and 'clipAllWeights' instead.");
}

ClipNormSettings::ClipNormSettings(const std::vector<TensorId> &weightIds_,
                                   float maxNorm_,
                                   Mode mode_)
    : weightIds(weightIds_), maxNorm(maxNorm_), mode(mode_) {}

ClipNormSettings
ClipNormSettings::clipWeights(const std::vector<TensorId> &weightIds_,
                              float maxNorm_) {
  return {weightIds_, maxNorm_, Mode::ClipSpecifiedWeights};
}

ClipNormSettings ClipNormSettings::clipAllWeights(float maxNorm_) {
  return {{}, maxNorm_, Mode::ClipAllWeights};
}

const std::vector<TensorId> &ClipNormSettings::getWeightIds() const {
  if (mode == Mode::ClipSpecifiedWeights) {
    return weightIds;
  } else {
    throw error("ClipNormSettings.getWeightIds should only be called if "
                "ClipNormSettings.getMode returns Mode::ClipSpecifiedWeights.");
  }
}

float ClipNormSettings::getMaxNorm() const { return maxNorm; }

ClipNormSettings::Mode ClipNormSettings::getMode() const { return mode; }

bool ClipNormSettings::operator==(const ClipNormSettings &other) const {
  if (weightIds != other.weightIds) {
    return false;
  }

  if (!isAlmostEqual(maxNorm, other.maxNorm)) {
    return false;
  }

  if (mode != other.mode) {
    return false;
  }

  return true;
}

bool ClipNormSettings::operator!=(const ClipNormSettings &other) const {
  return !(*this == other);
}

} // namespace popart
