// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <memory>
#include <popart/op/reduceprod.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/tensor.hpp>

namespace popart {

ReduceProdOp::ReduceProdOp(const OperatorIdentifier &_opid,
                           const std::vector<int64_t> &axes_,
                           const int64_t keepdims_,
                           const Op::Settings &settings_)
    : ReduceOp(_opid, axes_, keepdims_, settings_) {}

std::unique_ptr<Op> ReduceProdOp::clone() const {
  return std::make_unique<ReduceProdOp>(*this);
}

std::vector<std::unique_ptr<Op>> ReduceProdOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> result;
  result.emplace_back(
      std::make_unique<ReduceProdGradOp>(*this, backward_shape));
  return result;
}

ReduceProdGradOp::ReduceProdGradOp(const ReduceProdOp &fwdOp,
                                   const Shape &backward_shape_)
    : ReduceGradOp(Onnx::GradOperators::ReduceProdGrad,
                   fwdOp,
                   backward_shape_) {}

std::unique_ptr<Op> ReduceProdGradOp::clone() const {
  return std::make_unique<ReduceProdGradOp>(*this);
}

const std::vector<GradInOutMapper> &ReduceProdGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getInIndex(), ReduceProdOp::getOutIndex(), GradOpInType::GradOut},
      {getFwdInInIndex(), ReduceProdOp::getInIndex(), GradOpInType::In}};
  return inInfo;
}

namespace {

static OpDefinition::DataTypes T = {DataType::UINT32,
                                    DataType::UINT64,
                                    DataType::INT32,
                                    DataType::INT64,
                                    DataType::FLOAT16,
                                    DataType::FLOAT};

static OpDefinition reduceProdOpDef(
    {OpDefinition::Inputs({{"data", T}}),
     OpDefinition::Outputs({{"reduced", T}}),
     OpDefinition::Attributes({{"axes", {"*"}}, {"keepdims", {"*"}}})});

static OpCreator<ReduceProdOp> ReduceProdOpCreator(
    OpDefinitions({{Onnx::Operators::ReduceProd_1, reduceProdOpDef},
                   {Onnx::Operators::ReduceProd_11, reduceProdOpDef}}),
    [](const OperatorIdentifier &_opid,
       const Op::Settings &settings,
       const Attributes &attr) -> std::unique_ptr<Op> {
      int64_t keepdims = attr.getAttribute<Attributes::Int>("keepdims", 1);
      std::vector<int64_t> axes =
          attr.getAttribute<Attributes::Ints>("axes", {});

      return std::unique_ptr<Op>(
          new ReduceProdOp(_opid, axes, keepdims, settings));
    },
    true);
} // namespace

} // namespace popart
