// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <memory>
#include <popart/ir.hpp>
#include <popart/op/collectives/replicatedreducescatter.hpp>
#include <popart/opmanager.hpp>
#include <popart/region.hpp>
#include <popart/tensor.hpp>
#include <popart/util.hpp>

namespace popart {

ReplicatedReduceScatterOp::ReplicatedReduceScatterOp(
    const OperatorIdentifier &opid,
    const Op::Settings &settings)
    : CollectivesBaseOp(opid, settings) {}

std::unique_ptr<Op> ReplicatedReduceScatterOp::clone() const {
  return std::make_unique<ReplicatedReduceScatterOp>(*this);
}

void ReplicatedReduceScatterOp::setup() {

  const auto &inInfo_ = inInfo(getInIndex());

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
