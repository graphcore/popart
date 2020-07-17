// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <memory>
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
