#include <neuralnet/loss.hpp>

namespace neuralnet {
std::map<std::string, eLoss> initLossMap() {
  return {{"NLL", eLoss::NLL}, {"L1", eLoss::L1}};
}

const std::map<std::string, eLoss> &lossMap() {
  static std::map<std::string, eLoss> m = initLossMap();
  return m;
}
} // namespace neuralnet
