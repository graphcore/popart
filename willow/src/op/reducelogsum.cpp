// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <popart/op/reducelogsum.hpp>
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

ReduceLogSumOp::ReduceLogSumOp(
    const OperatorIdentifier &_opid,
    const nonstd::optional<std::vector<int64_t>> &axes_,
    const int64_t keepdims_,
    const Op::Settings &settings_)
    : ReduceOp(_opid, axes_, keepdims_, settings_) {}

std::unique_ptr<Op> ReduceLogSumOp::clone() const {
  return std::make_unique<ReduceLogSumOp>(*this);
}

std::vector<std::unique_ptr<Op>> ReduceLogSumOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> result;
  result.emplace_back(
      std::make_unique<ReduceLogSumGradOp>(*this, backward_shape));
  return result;
}

ReduceLogSumGradOp::ReduceLogSumGradOp(const ReduceLogSumOp &fwdOp,
                                       const Shape &backward_shape_)
    : ReduceGradOp(Onnx::GradOperators::ReduceLogSumGrad,
                   fwdOp,
                   backward_shape_) {}

std::unique_ptr<Op> ReduceLogSumGradOp::clone() const {
  return std::make_unique<ReduceLogSumGradOp>(*this);
}

const std::vector<GradInOutMapper> &ReduceLogSumGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getInIndex(), ReduceLogSumGradOp::getOutIndex(), GradOpInType::GradOut},
      {getFwdOutInIndex(),
       ReduceLogSumGradOp::getOutIndex(),
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

static OpDefinition reduceLogSumOpDef(
    {OpDefinition::Inputs({{"data", T}}),
     OpDefinition::Outputs({{"reduced", T}}),
     OpDefinition::Attributes({{"axes", {"*"}}, {"keepdims", {"*"}}})});

static OpCreator<ReduceLogSumOp> ReduceLogSumOpCreator(
    OpDefinitions({{Onnx::Operators::ReduceLogSum_1, reduceLogSumOpDef},
                   {Onnx::Operators::ReduceLogSum_11, reduceLogSumOpDef}}),
    [](const OpCreatorInfo &info) {
      int64_t keepdims =
          info.attributes.getAttribute<Attributes::Int>("keepdims", 1);
      nonstd::optional<std::vector<int64_t>> axes;
      if (info.attributes.hasAttribute("axes")) {
        axes = info.attributes.getAttribute<Attributes::Ints>("axes");
      }

      return std::unique_ptr<Op>(
          new ReduceLogSumOp(info.opid, axes, keepdims, info.settings));
    },
    true);
} // namespace

} // namespace popart
