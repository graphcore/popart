#include <sstream>
#include <willow/error.hpp>
#include <willow/loss.hpp>

namespace willow {

std::map<std::string, eLoss> initLossMap() {
  return {{"NLL", eLoss::NLL}, {"L1", eLoss::L1}};
}

const std::map<std::string, eLoss> &lossMap() {
  static std::map<std::string, eLoss> m = initLossMap();
  return m;
}

int Loss::input_size() const { return static_cast<int>(input_.size()); }

const TensorId &Loss::input(int i) const { return input_.at(i); }

int Loss::output_size() const { return 1; }
const TensorId &Loss::output(int i) const {
  if (i != 0) {
    throw error("only 1 loss output");
  }
  return output_;
}

Loss::Loss(const std::vector<TensorId> &in_, TensorId out_)
    : input_(in_), output_(out_) {}

} // namespace willow
