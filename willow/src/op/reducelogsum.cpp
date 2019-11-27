#include <algorithm>
#include <memory>
#include <popart/op/reducelogsum.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/tensor.hpp>

namespace popart {

ReduceLogSumOp::ReduceLogSumOp(const OperatorIdentifier &_opid,
                               const std::vector<int64_t> &axes_,
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
      {getInIndex(), ReduceLogSumGradOp::getOutIndex(), GradOpInType::GRADOUT},
      {getFwdOutInIndex(),
       ReduceLogSumGradOp::getOutIndex(),
       GradOpInType::OUT}};

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
    [](const OperatorIdentifier &_opid,
       const Op::Settings &settings,
       const Attributes &attr) -> std::unique_ptr<Op> {
      int64_t keepdims = attr.getAttribute<Attributes::Int>("keepdims", 1);
      std::vector<int64_t> axes =
          attr.getAttribute<Attributes::Ints>("axes", {});

      return std::unique_ptr<Op>(
          new ReduceLogSumOp(_opid, axes, keepdims, settings));
    },
    true);
} // namespace

} // namespace popart
