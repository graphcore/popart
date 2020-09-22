// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <memory>
#include <sstream>

#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/l1.hpp>
#include <popart/op/mean.hpp>
#include <popart/op/sum.hpp>
#include <popart/opmanager.hpp>
#include <popart/optimizer.hpp>
#include <popart/tensor.hpp>

namespace popart {

std::unique_ptr<Op> L1Op::clone() const {
  return std::make_unique<L1Op>(*this);
}

std::vector<std::unique_ptr<Op>> L1Op::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(std::make_unique<L1GradOp>(*this));
  return upops;
}

L1Op::L1Op(const OperatorIdentifier &_opid,
           const float lambda_,
           const ReductionType reduction_,
           const Op::Settings &settings_)
    : LossOp(_opid, settings_), lambda(lambda_), reduction(reduction_) {}

void L1GradOp::setup() {
  // gradient of input has same shape as input to L1
  outInfo(getOutIndex()) = inInfo(getFwdActInIndex());
}

void L1Op::setup() {
  TensorInfo info0 = inInfo(getInIndex());
  if (info0.rank() == 0) {
    throw error("L1Op not valid for rank-0 tensor (scalar)");
  }

  Shape outShape({});
  if (getReductionType() == ReductionType::NoReduction) {
    outShape = info0.shape();
  }

  outInfo(getOutIndex()).set(info0.dataType(), outShape);
}

std::map<TensorId, std::vector<TensorId>>
L1Op::shard(const std::map<TensorId, std::vector<TensorId>> &inputs) {
  std::map<TensorId, std::vector<TensorId>> outputs = Op::shard(inputs);

  switch (getReductionType()) {
  case ReductionType::Sum: {
    auto sumOpUp = std::make_unique<SumOp>(Onnx::Operators::Sum_8, settings);
    Op *sumOp    = sumOpUp.get();
    getGraph().moveIntoGraph(std::move(sumOpUp));
    auto l1OutId = outId(L1Op::getOutIndex());
    disconnectOutTensor(outTensor(L1Op::getOutIndex()));

    InIndex i = 0;
    for (auto l1ShardOutId : outputs[l1OutId]) {
      sumOp->connectInTensor(i, l1ShardOutId);
      ++i;
    }

    sumOp->connectOutTensor(SumOp::getOutIndex(), l1OutId);
    sumOp->setup();
    outputs[l1OutId] = {l1OutId};
    break;
  }
  case ReductionType::Mean: {
    auto meanOpUp = std::make_unique<MeanOp>(Onnx::Operators::Mean_8, settings);
    Op *meanOp    = meanOpUp.get();
    getGraph().moveIntoGraph(std::move(meanOpUp));
    auto l1OutId = outId(L1Op::getOutIndex());
    disconnectOutTensor(outTensor(L1Op::getOutIndex()));

    InIndex i = 0;
    for (auto l1ShardOutId : outputs[l1OutId]) {
      meanOp->connectInTensor(i, l1ShardOutId);
      ++i;
    }

    meanOp->connectOutTensor(MeanOp::getOutIndex(), l1OutId);
    meanOp->setup();
    outputs[l1OutId] = {l1OutId};
    break;
  }
  case ReductionType::NoReduction:
  default:
    break;
  }

  return outputs;
}

L1GradOp::L1GradOp(const L1Op &op_)
    : Op(Onnx::CustomGradOperators::L1Grad, op_.getSettings()),
      lambda(op_.getLambda()), reduction(op_.getReductionType()) {}

std::unique_ptr<Op> L1GradOp::clone() const {
  return std::make_unique<L1GradOp>(*this);
}

const std::vector<GradInOutMapper> &L1GradOp::gradInputInfo() const {
  // input at index 0 of this grad op is the input at index 0 of the L1
  // non-grad op.
  // input at index 1 of this grad op is the gradient of the output at index 0
  // of the L1 non-grad op.
  static const std::vector<GradInOutMapper> inInfo = {
      {getFwdActInIndex(), L1Op::getInIndex(), GradOpInType::In},
      {getGradInIndex(), L1Op::getOutIndex(), GradOpInType::GradOut}};
  return inInfo;
}

const std::map<int, int> &L1GradOp::gradOutToNonGradIn() const {
  // grad-op's (only) output corresponds to op's (only) input.
  static const std::map<int, int> outInfo = {
      {getOutIndex(), L1Op::getInIndex()}};
  return outInfo;
}

namespace {

static OpDefinition::DataTypes T = {DataType::FLOAT16,
                                    DataType::FLOAT,
                                    DataType::INT32,
                                    DataType::UINT32};

static OpDefinition l1lossOpDef(
    {OpDefinition::Inputs({{"A", T}}),
     OpDefinition::Outputs({{"B", T}}),
     OpDefinition::Attributes({{"lambda", {"*"}}, {"reduction", {"*"}}})});

static OpCreator<L1Op> l1lossOpCreator(
    OpDefinitions({{Onnx::CustomOperators::L1, l1lossOpDef}}),
    [](const OpCreatorInfo &info) {
      float lambda = info.attributes.getAttribute<Attributes::Float>("lambda");
      std::string reductionStr =
          info.attributes.getAttribute<Attributes::String>("reduction");
      ReductionType reduction = LossOp::reductionTypeFromString(reductionStr);
      return std::unique_ptr<L1Op>(
          new L1Op(info.opid, lambda, reduction, info.settings));
    },
    true);

} // namespace

} // namespace popart
