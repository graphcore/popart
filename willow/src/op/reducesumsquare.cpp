// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <memory>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/collectives/replicatedallreduce.hpp>
#include <popart/op/reducesumsquare.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/tensor.hpp>

namespace popart {

ReduceSumSquareOp::ReduceSumSquareOp(
    const OperatorIdentifier &_opid,
    const nonstd::optional<std::vector<int64_t>> &axes_,
    const int64_t keepdims_,
    const Op::Settings &settings_)
    : ReduceOp(_opid, axes_, keepdims_, settings_) {}

std::unique_ptr<Op> ReduceSumSquareOp::clone() const {
  return std::make_unique<ReduceSumSquareOp>(*this);
}

std::vector<std::unique_ptr<Op>> ReduceSumSquareOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> result;
  result.emplace_back(
      std::make_unique<ReduceSumSquareGradOp>(*this, backward_shape));
  return result;
}

ReplicatedTensorShardingIndices
ReduceSumSquareOp::getReplicatedTensorShardingIndices() const {
  return {{{ReduceSumSquareOp::getInIndex()}, {}}};
}

void ReduceSumSquareOp::configureForReplicatedTensorSharding(
    ReplicatedTensorShardingIndices indices,
    CommGroup shardingDomain) {
  if (indices == getReplicatedTensorShardingIndices()) {
    auto out = outTensor(ReduceSumSquareOp::getOutIndex());

    // Make sure reduction is only added once
    auto consumers = out->consumers.getOps();
    if (!std::any_of(consumers.begin(), consumers.end(), [](Op *op) {
          return dynamic_cast<ReplicatedAllReduceOp *>(op) ||
                 dynamic_cast<ReplicatedAllReduceInplaceOp *>(op);
        })) {

      TensorId intoReduceId =
          getGraph().getIr().createIntermediateTensorId(out->id);

      disconnectOutTensor(out);
      createAndConnectOutTensor(ReplicatedAllReduceOp::getOutIndex(),
                                intoReduceId);
      setup();

      auto reduceOp = getGraph().createOp<ReplicatedAllReduceInplaceOp>(
          Onnx::CustomOperators::ReplicatedAllReduceInplace,
          CollectiveOperator::Add,
          shardingDomain,
          settings);

      reduceOp->connectInTensor(ReplicatedAllReduceInplaceOp::getInIndex(),
                                intoReduceId);
      reduceOp->connectOutTensor(ReplicatedAllReduceInplaceOp::getOutIndex(),
                                 out->id);

      reduceOp->setup();
    }
  } else {
    throw error("ReduceSumSquareOp::configureForReplicatedTensorSharding "
                "Unexpected input indices.");
  }
}

ReduceSumSquareGradOp::ReduceSumSquareGradOp(const ReduceSumSquareOp &fwdOp,
                                             const Shape &backward_shape_)
    : ReduceGradOp(Onnx::GradOperators::ReduceSumSquareGrad,
                   fwdOp,
                   backward_shape_) {}

std::unique_ptr<Op> ReduceSumSquareGradOp::clone() const {
  return std::make_unique<ReduceSumSquareGradOp>(*this);
}

const std::vector<GradInOutMapper> &
ReduceSumSquareGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getInIndex(), ReduceSumSquareOp::getOutIndex(), GradOpInType::GradOut},
      {getFwdInInIndex(), ReduceSumSquareOp::getInIndex(), GradOpInType::In}};

  return inInfo;
}

namespace {

static OpDefinition::DataTypes T = {DataType::UINT32,
                                    DataType::UINT64,
                                    DataType::INT32,
                                    DataType::INT64,
                                    DataType::FLOAT16,
                                    DataType::FLOAT};

static OpDefinition reduceSumSquareOpDef(
    {OpDefinition::Inputs({
         {"data", T},
     }),
     OpDefinition::Outputs({{"reduced", T}}),
     OpDefinition::Attributes({{"axes", {"*"}}, {"keepdims", {"*"}}})});

static OpCreator<ReduceSumSquareOp> ReduceSumSquareOpCreator(
    OpDefinitions({{Onnx::Operators::ReduceSumSquare_1, reduceSumSquareOpDef},
                   {Onnx::Operators::ReduceSumSquare_11,
                    reduceSumSquareOpDef}}),
    [](const OpCreatorInfo &info) {
      int64_t keepdims =
          info.attributes.getAttribute<Attributes::Int>("keepdims", 1);
      nonstd::optional<std::vector<int64_t>> axes;
      if (info.attributes.hasAttribute("axes")) {
        axes = info.attributes.getAttribute<Attributes::Ints>("axes");
      }

      return std::unique_ptr<Op>(
          new ReduceSumSquareOp(info.opid, axes, keepdims, info.settings));
    },
    true);
} // namespace

} // namespace popart
