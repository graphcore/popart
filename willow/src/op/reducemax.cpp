// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <memory>
#include <popart/op/reducemax.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/tensor.hpp>

namespace popart {

ReduceMaxOp::ReduceMaxOp(const OperatorIdentifier &_opid,
                         const boost::optional<std::vector<int64_t>> &axes_,
                         const int64_t keepdims_,
                         const Op::Settings &settings_)
    : ReduceOp(_opid, axes_, keepdims_, settings_) {}

std::unique_ptr<Op> ReduceMaxOp::clone() const {
  return std::make_unique<ReduceMaxOp>(*this);
}

std::vector<std::unique_ptr<Op>> ReduceMaxOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> result;
  result.emplace_back(std::make_unique<ReduceMaxGradOp>(*this, backward_shape));
  return result;
}

ReduceMaxGradOp::ReduceMaxGradOp(const ReduceMaxOp &fwdOp,
                                 const Shape &backward_shape_)
    : ReduceGradOp(Onnx::GradOperators::ReduceMaxGrad, fwdOp, backward_shape_) {
}

std::unique_ptr<Op> ReduceMaxGradOp::clone() const {
  return std::make_unique<ReduceMaxGradOp>(*this);
}

const std::vector<GradInOutMapper> &ReduceMaxGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getInIndex(), ReduceMaxOp::getOutIndex(), GradOpInType::GradOut},
      {getFwdInInIndex(), ReduceMaxOp::getInIndex(), GradOpInType::In},
      {getFwdOutInIndex(), ReduceMaxOp::getOutIndex(), GradOpInType::Out}};
  return inInfo;
}

namespace {

static OpDefinition::DataTypes T = {DataType::UINT32,
                                    DataType::UINT64,
                                    DataType::INT32,
                                    DataType::INT64,
                                    DataType::FLOAT16,
                                    DataType::FLOAT};

static OpDefinition reduceMaxOpDef(
    {OpDefinition::Inputs({{"data", T}}),
     OpDefinition::Outputs({{"reduced", T}}),
     OpDefinition::Attributes({{"axes", {"*"}}, {"keepdims", {"*"}}})});

static OpCreator<ReduceMaxOp> reduceMaxOpCreator(
    OpDefinitions({{Onnx::Operators::ReduceMax_1, reduceMaxOpDef},
                   {Onnx::Operators::ReduceMax_11, reduceMaxOpDef}}),
    [](const OperatorIdentifier &_opid,
       const Op::Settings &settings,
       const Attributes &attr) -> std::unique_ptr<Op> {
      int64_t keepdims = attr.getAttribute<Attributes::Int>("keepdims", 1);
      boost::optional<std::vector<int64_t>> axes;
      if (attr.hasAttribute("axes")) {
        axes = attr.getAttribute<Attributes::Ints>("axes");
      }

      return std::unique_ptr<Op>(
          new ReduceMaxOp(_opid, axes, keepdims, settings));
    },
    true);
} // namespace

} // namespace popart
