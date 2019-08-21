#include <sstream>
#include <popart/error.hpp>
#include <popart/op/loss.hpp>

namespace popart {

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

ReductionType Loss::getReductionType() const { return reduction_type_; }

Loss::Loss(const std::vector<TensorId> &in_, TensorId out_, ReductionType rt_)
    : input_(in_), output_(out_), reduction_type_(rt_) {}

int64_t Loss::getVirtualGraphId() const {
  if (!hasVirtualGraphId()) {
    throw error(
        "Cannot return vGraphId for Loss {}. It has not had this attribute set",
        input_);
  }
  return *vgraphId;
}

bool Loss::hasVirtualGraphId() const {
  if (vgraphId) {
    return true;
  } else {
    return false;
  }
}

LossOp::LossOp(const OperatorIdentifier &_opid, const Op::Settings &settings_)
    : Op(_opid, settings_) {}

LossOp::LossOp(const Op &op) : Op(op) {}

bool LossOp::isLossOp() const { return true; }

} // namespace popart
