// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <memory>
#include <popart/ir.hpp>
#include <popart/op/collectives/replicatedallreduce.hpp>
#include <popart/opmanager.hpp>
#include <popart/region.hpp>
#include <popart/tensor.hpp>
#include <popart/util.hpp>

namespace popart {

ReplicatedAllReduceOp::ReplicatedAllReduceOp(const OperatorIdentifier &_opid,
                                             const Op::Settings &settings_)
    : CollectivesBaseOp(_opid, settings_) {}

std::unique_ptr<Op> ReplicatedAllReduceOp::clone() const {
  return std::make_unique<ReplicatedAllReduceOp>(*this);
}

std::unique_ptr<Op> ReplicatedAllReduceOp::getInplaceVariant(
    const OperatorIdentifier &operator_id) const {
  if (operator_id == Onnx::CustomOperators::ReplicatedAllReduceInplace) {
    return std::make_unique<ReplicatedAllReduceInplaceOp>(*this);
  }
  // catch remaining cases and throw an error
  return Op::getInplaceVariant(operator_id);
}

void ReplicatedAllReduceOp::setup() {
  outInfo(getOutIndex()) = inInfo(getInIndex());
}

ReplicatedAllReduceInplaceOp::ReplicatedAllReduceInplaceOp(
    const OperatorIdentifier &_opid,
    const Op::Settings &settings_)
    : ReplicatedAllReduceOp(_opid, settings_) {}

ReplicatedAllReduceInplaceOp::ReplicatedAllReduceInplaceOp(
    const ReplicatedAllReduceOp &rop)
    : ReplicatedAllReduceInplaceOp(
          Onnx::CustomOperators::ReplicatedAllReduceInplace,
          rop.getSettings()) {}

view::Regions ReplicatedAllReduceInplaceOp::modifies(InIndex index) const {
  if (index == getInIndex()) {
    return {view::Region::getFull(inShape(index))};
  } else {
    throw error("Invalid index passed to modifies");
  }
}

view::Regions ReplicatedAllReduceInplaceOp::aliases(InIndex in,
                                                    OutIndex out) const {
  if (in != out) {
    throw error("In index and out index are uneqal");
  }
  if (in == getInIndex()) {
    return {view::Region::getFull(inShape(in))};
  } else {
    throw error("Invalid index passed to aliases");
  }
}

std::unique_ptr<Op> ReplicatedAllReduceInplaceOp::clone() const {
  return std::make_unique<ReplicatedAllReduceInplaceOp>(*this);
}

void ReplicatedAllReduceInplaceOp::setup() {
  outInfo(getOutIndex()) = inInfo(getInIndex());
}

static OpDefinition::DataTypes T = {DataType::FLOAT,
                                    DataType::FLOAT16,
                                    DataType::INT32,
                                    DataType::UINT32};

static OpDefinition ReplicatedAllReduceOpDef({OpDefinition::Inputs({{"X", T}}),
                                              OpDefinition::Outputs({{"Y", T}}),
                                              OpDefinition::Attributes({})});

static OpCreator<ReplicatedAllReduceOp> ReplicatedAllReduceOpCreator(
    OpDefinitions({{Onnx::CustomOperators::ReplicatedAllReduce,
                    ReplicatedAllReduceOpDef}}),
    [](const OpCreatorInfo &info) {
      return std::unique_ptr<ReplicatedAllReduceOp>(
          new ReplicatedAllReduceOp(info.opid, info.settings));
    },
    true);

static OpDefinition
    ReplicatedAllReduceInplaceOpDef({OpDefinition::Inputs({{"X", T}}),
                                     OpDefinition::Outputs({{"Y", T}}),
                                     OpDefinition::Attributes({})});

static OpCreator<ReplicatedAllReduceInplaceOp>
    ReplicatedAllReduceInplaceOpCreator(
        OpDefinitions({{Onnx::CustomOperators::ReplicatedAllReduceInplace,
                        ReplicatedAllReduceInplaceOpDef}}),
        [](const OpCreatorInfo &info) {
          return std::unique_ptr<ReplicatedAllReduceInplaceOp>(
              new ReplicatedAllReduceInplaceOp(info.opid, info.settings));
        },
        true);

} // namespace popart
