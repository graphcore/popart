#ifndef GUARD_NEURALNET_ENIGMATIC_HPP
#define GUARD_NEURALNET_ENIGMATIC_HPP

#include <willow/names.hpp>

#pragma clang diagnostic push // start ignoring warnings
#pragma clang diagnostic ignored "-Weverything"
#include <poplar/OptionFlags.hpp>
#pragma clang diagnostic pop // stop ignoring warnings

// taken directly from enigma, commit
// Date:   Fri Oct 19 13:54:14 2018 +0100

namespace willow {
namespace popx {
namespace enigma {

enum class WeightUpdateMethod { AMP, AUTO };

enum class Pass {
  NONE,
  INFERENCE_FWD,
  TRAINING_FWD,
  TRAINING_BWD,
  TRAINING_WU,
  FC_INFERENCE_FWD,
  FC_TRAINING_FWD,
  FC_TRAINING_BWD,
  FC_TRAINING_WU
};

/** Options to control the implementation of convolutions used **/
struct ConvOptions {
  WeightUpdateMethod weightUpdateMethod = WeightUpdateMethod::AUTO;
  bool useWinograd                      = false;
  unsigned winogradPatchSize            = 4;
  unsigned tempMemoryBudget             = 0;
  /// The pass this layer corresponds to
  Pass pass = Pass::NONE;
};

enum class DeviceType { Cpu, IpuModel, Sim, Hw };

poplar::OptionFlags toPoplibsConvOptions(const ConvOptions &options);

} // namespace enigma
} // namespace popx
} // namespace willow

#endif
