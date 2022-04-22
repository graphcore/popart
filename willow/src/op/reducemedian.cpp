// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <popart/op/reducemedian.hpp>
#include <popart/opmanager.hpp>

#include "popart/attributes.hpp"
#include "popart/datatype.hpp"
#include "popart/graphcoreoperators.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/op/reduce.hpp"
#include "popart/tensorinfo.hpp"
#include "popart/vendored/optional.hpp"

namespace popart {
struct OperatorIdentifier;

ReduceMedianOp::ReduceMedianOp(
    const OperatorIdentifier &opid,
    const nonstd::optional<std::vector<int64_t>> &axes,
    int64_t keepdims,
    const Op::Settings &settings)
    : ReduceOp(opid, axes, keepdims, settings) {}

std::unique_ptr<Op> ReduceMedianOp::clone() const {
  return std::make_unique<ReduceMedianOp>(*this);
}

void ReduceMedianOp::setup() {
  ReduceOp::setup();
  outInfo(getIndicesOutIndex()) = {DataType::INT32,
                                   outInfo(getOutIndex()).shape()};
}

std::vector<std::unique_ptr<Op>> ReduceMedianOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> result;
  result.emplace_back(
      std::make_unique<ReduceMedianGradOp>(*this, backward_shape));
  return result;
}

bool ReduceMedianOp::canBeReplacedByIdentity() const {
  // Make sure the op is never replaced by identity as callers expect two
  // outputs and identity only has one.
  return false;
}

ReduceMedianGradOp::ReduceMedianGradOp(const ReduceMedianOp &fwd_op,
                                       const Shape &backward_shape)
    : ReduceGradOp(Onnx::CustomGradOperators::ReduceMedianGrad,
                   fwd_op,
                   backward_shape) {}

std::unique_ptr<Op> ReduceMedianGradOp::clone() const {
  return std::make_unique<ReduceMedianGradOp>(*this);
}

const std::vector<GradInOutMapper> &ReduceMedianGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      // Gradient from the top for the median values (0).
      {getInIndex(), ReduceMedianOp::getOutIndex(), GradOpInType::GradOut},
      // Indices computed during the forward pass (1).
      {getIndicesInIndex(),
       ReduceMedianOp::getIndicesOutIndex(),
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

static OpDefinition reduceMedianOpDef(
    {OpDefinition::Inputs({{"data", T}}),
     OpDefinition::Outputs({{"reduced", T}, {"indices", {DataType::INT32}}}),
     OpDefinition::Attributes({{"axes", {"*"}}, {"keepdims", {"*"}}})});

static OpCreator<ReduceMedianOp> ReduceMedianOpCreator(
    OpDefinitions({{Onnx::AiGraphcore::OpSet1::ReduceMedian,
                    reduceMedianOpDef}}),
    [](const OpCreatorInfo &info) {
      int64_t keepdims =
          info.attributes.getAttribute<Attributes::Int>("keepdims", 1);
      nonstd::optional<std::vector<int64_t>> axes;
      if (info.attributes.hasAttribute("axes")) {
        axes = info.attributes.getAttribute<Attributes::Ints>("axes");
      }
      return std::unique_ptr<Op>(
          new ReduceMedianOp(info.opid, axes, keepdims, info.settings));
    },
    true);
} // namespace

} // namespace popart
