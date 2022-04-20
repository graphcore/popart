// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <popart/op/reducemean.hpp>
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

ReduceMeanOp::ReduceMeanOp(const OperatorIdentifier &_opid,
                           const nonstd::optional<std::vector<int64_t>> &axes_,
                           const int64_t keepdims_,
                           const Op::Settings &settings_)
    : ReduceOp(_opid, axes_, keepdims_, settings_) {}

std::unique_ptr<Op> ReduceMeanOp::clone() const {
  return std::make_unique<ReduceMeanOp>(*this);
}

std::vector<std::unique_ptr<Op>> ReduceMeanOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> result;
  result.emplace_back(
      std::make_unique<ReduceMeanGradOp>(*this, backward_shape));
  return result;
}

ReduceMeanGradOp::ReduceMeanGradOp(const ReduceMeanOp &fwdOp,
                                   const Shape &backward_shape_)
    : ReduceGradOp(Onnx::GradOperators::ReduceMeanGrad,
                   fwdOp,
                   backward_shape_) {}

std::unique_ptr<Op> ReduceMeanGradOp::clone() const {
  return std::make_unique<ReduceMeanGradOp>(*this);
}

namespace {

static OpDefinition::DataTypes T = {DataType::UINT32,
                                    DataType::UINT64,
                                    DataType::INT32,
                                    DataType::INT64,
                                    DataType::FLOAT16,
                                    DataType::FLOAT};

static OpDefinition reduceMeanOpDef(
    {OpDefinition::Inputs({{"data", T}}),
     OpDefinition::Outputs({{"reduced", T}}),
     OpDefinition::Attributes({{"axes", {"*"}}, {"keepdims", {"*"}}})});

static OpCreator<ReduceMeanOp> ReduceMeanOpCreator(
    OpDefinitions({{Onnx::Operators::ReduceMean_1, reduceMeanOpDef},
                   {Onnx::Operators::ReduceMean_11, reduceMeanOpDef}}),
    [](const OpCreatorInfo &info) {
      int64_t keepdims =
          info.attributes.getAttribute<Attributes::Int>("keepdims", 1);
      nonstd::optional<std::vector<int64_t>> axes;
      if (info.attributes.hasAttribute("axes")) {
        axes = info.attributes.getAttribute<Attributes::Ints>("axes");
      }

      return std::unique_ptr<Op>(
          new ReduceMeanOp(info.opid, axes, keepdims, info.settings));
    },
    true);
} // namespace

} // namespace popart
