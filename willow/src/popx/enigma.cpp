#include <willow/popx/enigma.hpp>

namespace willow {
namespace popx {
namespace enigma {

poplar::OptionFlags toPoplibsConvOptions(const ConvOptions &options) {
  poplar::OptionFlags convOpt;
  switch (options.weightUpdateMethod) {
  case WeightUpdateMethod::AMP:
    convOpt.set("weightUpdateMethod", "AMP");
    break;
  case WeightUpdateMethod::AUTO:
    convOpt.set("weightUpdateMethod", "AUTO");
    break;
  }
  convOpt.set("useWinograd", options.useWinograd ? "true" : "false");
  std::stringstream s;
  s << options.winogradPatchSize;
  convOpt.set("winogradPatchSize", s.str());
  s.str({});
  // s << options.tempMemoryBudget;
  // convOpt.set("tempMemoryBudget", s.str());
  switch (options.pass) {
  case Pass::NONE:
    convOpt.set("pass", "NONE");
    break;
  case Pass::INFERENCE_FWD:
    convOpt.set("pass", "INFERENCE_FWD");
    break;
  case Pass::TRAINING_FWD:
    convOpt.set("pass", "TRAINING_FWD");
    break;
  case Pass::TRAINING_BWD:
    convOpt.set("pass", "TRAINING_BWD");
    break;
  case Pass::TRAINING_WU:
    convOpt.set("pass", "TRAINING_WU");
    break;
  case Pass::FC_INFERENCE_FWD:
    convOpt.set("pass", "FC_INFERENCE_FWD");
    break;
  case Pass::FC_TRAINING_FWD:
    convOpt.set("pass", "FC_TRAINING_FWD");
    break;
  case Pass::FC_TRAINING_BWD:
    convOpt.set("pass", "FC_TRAINING_BWD");
    break;
  case Pass::FC_TRAINING_WU:
    convOpt.set("pass", "FC_TRAINING_WU");
    break;
  }
  return convOpt;
}

} // namespace enigma
} // namespace popx
} // namespace willow
