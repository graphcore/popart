// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <popart/op/reducel1.hpp>
#include <popart/opmanager.hpp>

#include "popart/attributes.hpp"
#include "popart/datatype.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/op/reduce.hpp"
#include "popart/operators.hpp"
#include "popart/vendored/optional.hpp"

namespace popart {
struct OperatorIdentifier;

ReduceL1Op::ReduceL1Op(const OperatorIdentifier &_opid,
                       const nonstd::optional<std::vector<int64_t>> &axes_,
                       const int64_t keepdims_,
                       const Op::Settings &settings_)
    : ReduceOp(_opid, axes_, keepdims_, settings_) {}

std::unique_ptr<Op> ReduceL1Op::clone() const {
  return std::make_unique<ReduceL1Op>(*this);
}

std::vector<std::unique_ptr<Op>> ReduceL1Op::getGradOps() {
  std::vector<std::unique_ptr<Op>> result;
  result.emplace_back(std::make_unique<ReduceL1GradOp>(*this, backward_shape));
  return result;
}

ReduceL1GradOp::ReduceL1GradOp(const ReduceL1Op &fwdOp,
                               const Shape &backward_shape_)
    : ReduceGradOp(Onnx::GradOperators::ReduceL1Grad, fwdOp, backward_shape_) {}

std::unique_ptr<Op> ReduceL1GradOp::clone() const {
  return std::make_unique<ReduceL1GradOp>(*this);
}

const std::vector<GradInOutMapper> &ReduceL1GradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getInIndex(), ReduceL1GradOp::getOutIndex(), GradOpInType::GradOut},
      {getFwdInInIndex(), ReduceL1GradOp::getInIndex(), GradOpInType::In}};

  return inInfo;
}

namespace {

static OpDefinition::DataTypes T = {DataType::UINT32,
                                    DataType::UINT64,
                                    DataType::INT32,
                                    DataType::INT64,
                                    DataType::FLOAT16,
                                    DataType::FLOAT};

static OpDefinition reduceL1OpDef(
    {OpDefinition::Inputs({{"data", T}}),
     OpDefinition::Outputs({{"reduced", T}}),
     OpDefinition::Attributes({{"axes", {"*"}}, {"keepdims", {"*"}}})});

static OpCreator<ReduceL1Op> reduceL1OpCreator(
    OpDefinitions({{Onnx::Operators::ReduceL1_1, reduceL1OpDef},
                   {Onnx::Operators::ReduceL1_11, reduceL1OpDef}}),
    [](const OpCreatorInfo &info) {
      int64_t keepdims =
          info.attributes.getAttribute<Attributes::Int>("keepdims", 1);
      nonstd::optional<std::vector<int64_t>> axes;
      if (info.attributes.hasAttribute("axes")) {
        axes = info.attributes.getAttribute<Attributes::Ints>("axes");
      }

      return std::unique_ptr<Op>(
          new ReduceL1Op(info.opid, axes, keepdims, info.settings));
    },
    true);
} // namespace

} // namespace popart
