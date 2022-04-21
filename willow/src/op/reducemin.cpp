// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <memory>
#include <popart/op/reducemin.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/tensor.hpp>

namespace popart {

ReduceMinOp::ReduceMinOp(const OperatorIdentifier &_opid,
                         const nonstd::optional<std::vector<int64_t>> &axes_,
                         const int64_t keepdims_,
                         const Op::Settings &settings_)
    : ReduceOp(_opid, axes_, keepdims_, settings_) {}

std::unique_ptr<Op> ReduceMinOp::clone() const {
  return std::make_unique<ReduceMinOp>(*this);
}

std::vector<std::unique_ptr<Op>> ReduceMinOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> result;
  result.emplace_back(std::make_unique<ReduceMinGradOp>(*this, backward_shape));
  return result;
}

ReduceMinGradOp::ReduceMinGradOp(const ReduceMinOp &fwdOp,
                                 const Shape &backward_shape_)
    : ReduceGradOp(Onnx::GradOperators::ReduceMinGrad, fwdOp, backward_shape_) {
}

std::unique_ptr<Op> ReduceMinGradOp::clone() const {
  return std::make_unique<ReduceMinGradOp>(*this);
}

const std::vector<GradInOutMapper> &ReduceMinGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getInIndex(), ReduceMinOp::getOutIndex(), GradOpInType::GradOut},
      {getFwdInInIndex(), ReduceMinOp::getInIndex(), GradOpInType::In},
      {getFwdOutInIndex(), ReduceMinOp::getOutIndex(), GradOpInType::Out}};
  return inInfo;
}

namespace {

static OpDefinition::DataTypes T = {DataType::UINT32,
                                    DataType::UINT64,
                                    DataType::INT32,
                                    DataType::INT64,
                                    DataType::FLOAT16,
                                    DataType::FLOAT};

static OpDefinition reduceMinOpDef(
    {OpDefinition::Inputs({{"data", T}}),
     OpDefinition::Outputs({{"reduced", T}}),
     OpDefinition::Attributes({{"axes", {"*"}}, {"keepdims", {"*"}}})});

static OpCreator<ReduceMinOp> reduceMinOpCreator(
    OpDefinitions({{Onnx::Operators::ReduceMin_1, reduceMinOpDef},
                   {Onnx::Operators::ReduceMin_11, reduceMinOpDef}}),
    [](const OpCreatorInfo &info) {
      int64_t keepdims =
          info.attributes.getAttribute<Attributes::Int>("keepdims", 1);
      nonstd::optional<std::vector<int64_t>> axes;
      if (info.attributes.hasAttribute("axes")) {
        axes = info.attributes.getAttribute<Attributes::Ints>("axes");
      }

      return std::unique_ptr<Op>(
          new ReduceMinOp(info.opid, axes, keepdims, info.settings));
    },
    true);
} // namespace

} // namespace popart
