// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <memory>
#include <popart/ir.hpp>
#include <popart/op/collectives.hpp>
#include <popart/opmanager.hpp>
#include <popart/region.hpp>
#include <popart/tensor.hpp>
#include <popart/util.hpp>

namespace popart {

CollectivesBaseOp::CollectivesBaseOp(const OperatorIdentifier &opid,
                                     const Op::Settings &settings)
    : Op(opid, settings) {}

ReplicatedAllReduceOp::ReplicatedAllReduceOp(const OperatorIdentifier &opid,
                                             const Op::Settings &settings)
    : CollectivesBaseOp(opid, settings) {}

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
    const OperatorIdentifier &opid,
    const Op::Settings &settings)
    : CollectivesBaseOp(opid, settings) {}

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

ReplicatedReduceScatterOp::ReplicatedReduceScatterOp(
    const OperatorIdentifier &opid,
    const Op::Settings &settings)
    : CollectivesBaseOp(opid, settings) {}

std::unique_ptr<Op> ReplicatedReduceScatterOp::clone() const {
  return std::make_unique<ReplicatedReduceScatterOp>(*this);
}

void ReplicatedReduceScatterOp::setup() {

  const auto &inInfo_ = inInfo(getInIndex());
  if (inInfo_.rank() != 1) {
    throw error("ReduceScatter input is a rank {} tensor, but reduceScatter "
                "requires rank 1 tensor as input",
                inInfo_.rank());
  }

  const auto replicationFactor =
      getIr().getSessionOptions().replicatedGraphCount;
  int64_t nelms = inInfo_.nelms();

  // ceil(numElements / replicationFactor)
  auto outElms = (nelms + replicationFactor - 1) / replicationFactor;

  outInfo(getOutIndex()) = TensorInfo(inInfo_.dataType(), {outElms});
}

static OpDefinition::DataTypes T = {DataType::FLOAT,
                                    DataType::FLOAT16,
                                    DataType::INT32,
                                    DataType::UINT32};

static OpDefinition
    ReplicatedAllReduceInplaceOpDef({OpDefinition::Inputs({{"X", T}}),
                                     OpDefinition::Outputs({{"Y", T}}),
                                     OpDefinition::Attributes({})});

static OpCreator<ReplicatedAllReduceInplaceOp>
    ReplicatedAllReduceInplaceOpCreator(
        OpDefinitions({{Onnx::CustomOperators::ReplicatedAllReduceInplace,
                        ReplicatedAllReduceInplaceOpDef}}),
        [](const OperatorIdentifier &_opid,
           const Op::Settings &settings,
           const Attributes &attr) -> std::unique_ptr<Op> {
          return std::unique_ptr<ReplicatedAllReduceInplaceOp>(
              new ReplicatedAllReduceInplaceOp(_opid, settings));
        },
        true);

static OpDefinition ReplicatedAllReduceOpDef({OpDefinition::Inputs({{"X", T}}),
                                              OpDefinition::Outputs({{"Y", T}}),
                                              OpDefinition::Attributes({})});

static OpCreator<ReplicatedAllReduceOp> ReplicatedAllReduceOpCreator(
    OpDefinitions({{Onnx::CustomOperators::ReplicatedAllReduce,
                    ReplicatedAllReduceOpDef}}),
    [](const OperatorIdentifier &_opid,
       const Op::Settings &settings,
       const Attributes &attr = {}) -> std::unique_ptr<Op> {
      return std::unique_ptr<ReplicatedAllReduceOp>(
          new ReplicatedAllReduceOp(_opid, settings));
    },
    true);

static OpDefinition
    ReplicatedReduceScatterOpDef({OpDefinition::Inputs({{"X", T}}),
                                  OpDefinition::Outputs({{"Y", T}}),
                                  OpDefinition::Attributes({})});

static OpCreator<ReplicatedReduceScatterOp> ReplicatedReduceScatterOpCreator(
    OpDefinitions({{Onnx::CustomOperators::ReplicatedReduceScatter,
                    ReplicatedReduceScatterOpDef}}),
    [](const OperatorIdentifier &_opid,
       const Op::Settings &settings,
       const Attributes &attr = {}) -> std::unique_ptr<Op> {
      return std::unique_ptr<ReplicatedReduceScatterOp>(
          new ReplicatedReduceScatterOp(_opid, settings));
    },
    true);

} // namespace popart
