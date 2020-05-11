// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <sstream>
#include <popart/error.hpp>
#include <popart/op/loss.hpp>

namespace popart {

std::string LossOp::reductionTypeToString(ReductionType reduction) {
  if (reduction == ReductionType::Sum) {
    return "Sum";
  } else if (reduction == ReductionType::Mean) {
    return "Mean";
  } else {
    throw error("Unexpected ReductionType. Cannot convert to string");
  }
}

ReductionType LossOp::reductionTypeFromString(std::string reduction) {
  if (reduction == "Sum") {
    return ReductionType::Sum;
  } else if (reduction == "Mean") {
    return ReductionType::Mean;
  } else {
    throw error(
        "Unexpected ReductionType string, {}. Cannot convert to ReductionType",
        reduction);
  }
}

std::map<std::string, eLoss> initLossMap() {
  return {{"NLL", eLoss::NLL}, {"L1", eLoss::L1}, {"ID", eLoss::ID}};
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

bool Loss::hasPipelineStage() const { return pipelineStage_ != boost::none; }

PipelineStage Loss::getPipelineStage() const {
  if (!hasPipelineStage()) {
    throw error("Cannot return pipeline stage for Loss {}. It has not had this "
                "attribute set",
                input_);
  }
  return *pipelineStage_;
}

LossOp::LossOp(const OperatorIdentifier &_opid, const Op::Settings &settings_)
    : Op(_opid, settings_) {}

LossOp::LossOp(const Op &op) : Op(op) {}

bool LossOp::isLossOp() const { return true; }

} // namespace popart
