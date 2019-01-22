#include <sstream>
#include <poponnx/error.hpp>
#include <poponnx/op/loss.hpp>

namespace poponnx {

std::map<std::string, eLoss> initLossMap() {
  return {{"NLL", eLoss::NLL}, {"L1", eLoss::L1}};
}

const std::map<std::string, eLoss> &lossMap() {
  static std::map<std::string, eLoss> m = initLossMap();
  return m;
}

int Loss::input_size() const { return static_cast<int>(input_.size()); }

const TensorId &Loss::input(InIndex i) const { return input_.at(i); }

int Loss::output_size() const { return 1; }
const TensorId &Loss::output(OutIndex i) const {
  if (i != 0) {
    throw error("only 1 loss output");
  }
  return output_;
}

Loss::Loss(const std::vector<TensorId> &in_, TensorId out_)
    : input_(in_), output_(out_) {}

LossOp::LossOp(const OperatorIdentifier &_opid, const Op::Settings &settings_)
    : Op(_opid, settings_) {}

LossOp::LossOp(const Op &op) : Op(op) {}

bool LossOp::isLossOp() const { return true; }

} // namespace poponnx
