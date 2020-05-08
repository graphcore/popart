// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/collectives/replicatedallgather.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/tensor.hpp>

namespace popart {

ReplicatedAllGatherOp::ReplicatedAllGatherOp(const OperatorIdentifier &_opid,
                                             const Op::Settings &settings_)
    : CollectivesBaseOp(_opid, settings_) {}

ReplicatedAllGatherOp::ReplicatedAllGatherOp(const OperatorIdentifier &_opid,
                                             const Op::Settings &settings_,
                                             TensorInfo gatheredOutInfo_)
    : CollectivesBaseOp(_opid, settings_), gatheredOutInfo(gatheredOutInfo_) {}

std::unique_ptr<Op> ReplicatedAllGatherOp::clone() const {
  return std::make_unique<ReplicatedAllGatherOp>(*this);
}

void ReplicatedAllGatherOp::setup() {
  const auto replicationFactor =
      getIr().getSessionOptions().replicatedGraphCount;
  if (gatheredOutInfo.shape().empty()) {
    gatheredOutInfo = inInfo(ReplicatedAllGatherOp::getInIndex());
    Shape shape(1, replicationFactor * gatheredOutInfo.nelms());
    gatheredOutInfo.set(gatheredOutInfo.dataType(), shape);
  }
  outInfo(getOutIndex()) = gatheredOutInfo;
}

static OpDefinition::DataTypes T = {DataType::FLOAT,
                                    DataType::FLOAT16,
                                    DataType::INT32,
                                    DataType::UINT32};

static OpDefinition ReplicatedAllGatherOpDef({OpDefinition::Inputs({{"X", T}}),
                                              OpDefinition::Outputs({{"Y", T}}),
                                              OpDefinition::Attributes({})});

static OpCreator<ReplicatedAllGatherOp> ReplicatedAllGatherOpCreator(
    OpDefinitions({{Onnx::CustomOperators::ReplicatedAllGather,
                    ReplicatedAllGatherOpDef}}),
    [](const OperatorIdentifier &_opid,
       const Op::Settings &settings,
       const Attributes & = {}) -> std::unique_ptr<Op> {
      return std::unique_ptr<ReplicatedAllGatherOp>(
          new ReplicatedAllGatherOp(_opid, settings));
    },
    true);

} // namespace popart
