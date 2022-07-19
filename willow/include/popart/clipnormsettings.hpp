// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_CLIPNORMSETTINGS_HPP_
#define POPART_WILLOW_INCLUDE_POPART_CLIPNORMSETTINGS_HPP_
#include <string>
#include <vector>
#include <popart/names.hpp>

namespace popart {

/**
 * A data structure used to represent a maximum value constraint on
 * one or more weights. This is passed to the optimizer on construction.
 */
class ClipNormSettings {
public:
  // Mode for clip norm settings:
  //   ClipSpecifiedWeights - apply gradient clipping to the given weights.
  //   ClipAllWeights - apply gradient clipping on all the weights in the model.
  enum class Mode { ClipSpecifiedWeights, ClipAllWeights };

  /// DEPRECATED This will be removed from a future release.
  /// Constructor.
  /// \param weightIds_ The weight tensor IDs that this constraint
  ///     applies to.
  /// \param maxNorm_ The maximum permissible value.
  ClipNormSettings(const std::vector<TensorId> &weightIds_, float maxNorm_);

public:
  // static factory method to create a ClipNormSettings that specifies a vector
  // of weights to clip.
  static ClipNormSettings clipWeights(const std::vector<TensorId> &weightIds_,
                                      float maxNorm_);

  // static factor method to create a ClipNormSettings that specifies clipping
  // all weights in a model.
  static ClipNormSettings clipAllWeights(float maxNorm_);

  const std::vector<TensorId> &getWeightIds() const;
  float getMaxNorm() const;
  Mode getMode() const;

  bool operator==(const ClipNormSettings &) const;
  bool operator!=(const ClipNormSettings &other) const;

  // DEPRECATED This member will be made private in a future release.
  // The weightIds to perform gradient clipping on. This is only used if `mode`
  // is set to Mode::ClipSpecifiedWeights.
  std::vector<TensorId> weightIds;
  // DEPRECATED This member will be made private in a future release.
  float maxNorm;

private:
  // Hiding the constructors as the static factory methods should be used for
  // creation.
  ClipNormSettings() = default;
  ClipNormSettings(const std::vector<TensorId> &weightIds_,
                   float maxNorm_,
                   Mode mode_);

  Mode mode;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_CLIPNORMSETTINGS_HPP_
