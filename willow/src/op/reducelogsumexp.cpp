// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <memory>
#include <popart/op/reducelogsumexp.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/tensor.hpp>

namespace popart {

ReduceLogSumExpOp::ReduceLogSumExpOp(
    const OperatorIdentifier &_opid,
    const nonstd::optional<std::vector<int64_t>> &axes_,
    const int64_t keepdims_,
    const Op::Settings &settings_)
    : ReduceOp(_opid, axes_, keepdims_, settings_) {}

std::unique_ptr<Op> ReduceLogSumExpOp::clone() const {
  return std::make_unique<ReduceLogSumExpOp>(*this);
}

std::vector<std::unique_ptr<Op>> ReduceLogSumExpOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> result;
  result.emplace_back(
      std::make_unique<ReduceLogSumExpGradOp>(*this, backward_shape));
  return result;
}

ReduceLogSumExpGradOp::ReduceLogSumExpGradOp(const ReduceLogSumExpOp &fwdOp,
                                             const Shape &backward_shape_)
    : ReduceGradOp(Onnx::GradOperators::ReduceLogSumExpGrad,
                   fwdOp,
                   backward_shape_) {}

std::unique_ptr<Op> ReduceLogSumExpGradOp::clone() const {
  return std::make_unique<ReduceLogSumExpGradOp>(*this);
}

const std::vector<GradInOutMapper> &
ReduceLogSumExpGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getInIndex(),
       ReduceLogSumExpGradOp::getOutIndex(),
       GradOpInType::GradOut},
      {getFwdInInIndex(),
       ReduceLogSumExpGradOp::getInIndex(),
       GradOpInType::In},
      {getFwdOutInIndex(),
       ReduceLogSumExpGradOp::getOutIndex(),
       GradOpInType::Out}};

  return inInfo;
}

namespace {

static OpDefinition::DataTypes T = {DataType::UINT32,
                                    DataType::UINT64,
                                    DataType::INT32,
                                    DataType::INT64,
                                    DataType::FLOAT16,
                                    DataType::FLOAT};

static OpDefinition reduceLogSumExpOpDef(
    {OpDefinition::Inputs({{"data", T}}),
     OpDefinition::Outputs({{"reduced", T}}),
     OpDefinition::Attributes({{"axes", {"*"}}, {"keepdims", {"*"}}})});

static OpCreator<ReduceLogSumExpOp> ReduceLogSumExpOpCreator(
    OpDefinitions({{Onnx::Operators::ReduceLogSumExp_1, reduceLogSumExpOpDef},
                   {Onnx::Operators::ReduceLogSumExp_11,
                    reduceLogSumExpOpDef}}),
    [](const OpCreatorInfo &info) {
      int64_t keepdims =
          info.attributes.getAttribute<Attributes::Int>("keepdims", 1);
      nonstd::optional<std::vector<int64_t>> axes;
      if (info.attributes.hasAttribute("axes")) {
        axes = info.attributes.getAttribute<Attributes::Ints>("axes");
      }

      return std::unique_ptr<Op>(
          new ReduceLogSumExpOp(info.opid, axes, keepdims, info.settings));
    },
    true);
} // namespace

} // namespace popart
