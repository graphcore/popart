// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <memory>
#include <popart/ir.hpp>
#include <popart/op/dropoutbase.hpp>
#include <popart/opmanager.hpp>
#include <popart/tensor.hpp>

namespace popart {

DropoutBaseOp::DropoutBaseOp(const OperatorIdentifier &opid_,
                             float ratio_,
                             uint32_t seedModifier_,
                             const Op::Settings &settings_)
    : Op(opid_, settings_), ratio(ratio_), seedModifier(seedModifier_) {}

DropoutBaseOp::DropoutBaseOp(const OperatorIdentifier &opid_,
                             float ratio_,
                             const Op::Settings &settings_)
    : Op(opid_, settings_), ratio(ratio_), seedModifier(0) {
  updateSeedModifier();
}

// Dropout in testing mode can be replaced by the identity
bool DropoutBaseOp::canBeReplacedByIdentity() const {
  return (getIr().isTesting());
}

void DropoutBaseOp::updateSeedModifier() {
  seedModifier = getIr().getAndIncrementSeedModifier();
}

std::map<TensorId, std::vector<TensorId>>
DropoutBaseOp::shard(const std::map<TensorId, std::vector<TensorId>> &inputs) {
  auto outputs = Op::shard(inputs);
  for (auto shard_outs : outputs.begin()->second) {
    auto sharded_dropout = dynamic_cast<DropoutBaseOp *>(
        getIr().getTensor(shard_outs)->getProducer());
    // Fetch a unique seed modifier
    sharded_dropout->updateSeedModifier();
  }
  return outputs;
}

float DropoutBaseOp::validateRatioAttribute(const OpCreatorInfo &info) {
  float ratio = info.attributes.getAttribute<Attributes::Float>("ratio", 0.5f);
  // If invalid probability for ratio supplied, throw error.
  if (ratio <= float(0.) || ratio >= float(1.)) {
    throw error("{} ratio value {} is not valid. Please use a value in the "
                "interval (0,1)",
                info.opid,
                ratio);
  }
  return ratio;
}

} // namespace popart