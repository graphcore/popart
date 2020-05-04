// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <memory>
#include <popart/op/reducel2.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/tensor.hpp>

namespace popart {

ReduceL2Op::ReduceL2Op(const OperatorIdentifier &_opid,
                       const std::vector<int64_t> &axes_,
                       const int64_t keepdims_,
                       const Op::Settings &settings_)
    : ReduceOp(_opid, axes_, keepdims_, settings_) {}

std::unique_ptr<Op> ReduceL2Op::clone() const {
  return std::make_unique<ReduceL2Op>(*this);
}

std::vector<std::unique_ptr<Op>> ReduceL2Op::getGradOps() {
  std::vector<std::unique_ptr<Op>> result;
  result.emplace_back(std::make_unique<ReduceL2GradOp>(*this, backward_shape));
  return result;
}

ReduceL2GradOp::ReduceL2GradOp(const ReduceL2Op &fwdOp,
                               const Shape &backward_shape_)
    : ReduceGradOp(Onnx::GradOperators::ReduceL2Grad, fwdOp, backward_shape_) {}

std::unique_ptr<Op> ReduceL2GradOp::clone() const {
  return std::make_unique<ReduceL2GradOp>(*this);
}

const std::vector<GradInOutMapper> &ReduceL2GradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getInIndex(), ReduceL2GradOp::getOutIndex(), GradOpInType::GradOut},
      {getFwdInInIndex(), ReduceL2GradOp::getInIndex(), GradOpInType::In},
      {getFwdOutInIndex(), ReduceL2GradOp::getOutIndex(), GradOpInType::Out}};

  return inInfo;
}

namespace {

static OpDefinition::DataTypes T = {DataType::UINT32,
                                    DataType::UINT64,
                                    DataType::INT32,
                                    DataType::INT64,
                                    DataType::FLOAT16,
                                    DataType::FLOAT};

static OpDefinition reduceL2OpDef(
    {OpDefinition::Inputs({{"data", T}}),
     OpDefinition::Outputs({{"reduced", T}}),
     OpDefinition::Attributes({{"axes", {"*"}}, {"keepdims", {"*"}}})});

static OpCreator<ReduceL2Op> reduceL2OpCreator(
    OpDefinitions({{Onnx::Operators::ReduceL2_1, reduceL2OpDef},
                   {Onnx::Operators::ReduceL2_11, reduceL2OpDef}}),
    [](const OperatorIdentifier &_opid,
       const Op::Settings &settings,
       const Attributes &attr) -> std::unique_ptr<Op> {
      int64_t keepdims = attr.getAttribute<Attributes::Int>("keepdims", 1);
      std::vector<int64_t> axes =
          attr.getAttribute<Attributes::Ints>("axes", {});

      return std::unique_ptr<Op>(
          new ReduceL2Op(_opid, axes, keepdims, settings));
    },
    true);
} // namespace

} // namespace popart
