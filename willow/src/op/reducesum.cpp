// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <memory>
#include <popart/op/reducesum.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/tensor.hpp>

namespace popart {

ReduceSumOp::ReduceSumOp(const OperatorIdentifier &_opid,
                         const std::vector<int64_t> &axes_,
                         const int64_t keepdims_,
                         const Op::Settings &settings_)
    : ReduceOp(_opid, axes_, keepdims_, settings_) {}

std::unique_ptr<Op> ReduceSumOp::clone() const {
  return std::make_unique<ReduceSumOp>(*this);
}

std::vector<std::unique_ptr<Op>> ReduceSumOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> result;
  result.emplace_back(std::make_unique<ReduceSumGradOp>(*this, backward_shape));
  return result;
}

ReduceSumGradOp::ReduceSumGradOp(const ReduceSumOp &fwdOp,
                                 const Shape &backward_shape_)
    : ReduceGradOp(Onnx::GradOperators::ReduceSumGrad, fwdOp, backward_shape_) {
}

std::unique_ptr<Op> ReduceSumGradOp::clone() const {
  return std::make_unique<ReduceSumGradOp>(*this);
}

namespace {

static OpDefinition::DataTypes T = {DataType::UINT32,
                                    DataType::UINT64,
                                    DataType::INT32,
                                    DataType::INT64,
                                    DataType::FLOAT16,
                                    DataType::FLOAT};

static OpDefinition reduceSumOpDef(
    {OpDefinition::Inputs({{"data", T}}),
     OpDefinition::Outputs({{"reduced", T}}),
     OpDefinition::Attributes({{"axes", {"*"}}, {"keepdims", {"*"}}})});

static OpCreator<ReduceSumOp> reduceSumOpCreator(
    OpDefinitions({{Onnx::Operators::ReduceSum_1, reduceSumOpDef},
                   {Onnx::Operators::ReduceSum_11, reduceSumOpDef}}),
    [](const OperatorIdentifier &_opid,
       const Op::Settings &settings,
       const Attributes &attr) -> std::unique_ptr<Op> {
      int64_t keepdims = attr.getAttribute<Attributes::Int>("keepdims", 1);
      std::vector<int64_t> axes =
          attr.getAttribute<Attributes::Ints>("axes", {});

      return std::unique_ptr<Op>(
          new ReduceSumOp(_opid, axes, keepdims, settings));
    },
    true);
} // namespace

} // namespace popart
