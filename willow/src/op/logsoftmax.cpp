#include <memory>
#include <popart/error.hpp>
#include <popart/op/logsoftmax.hpp>
#include <popart/opmanager.hpp>
#include <popart/tensor.hpp>

namespace popart {

LogSoftmaxOp::LogSoftmaxOp(const OperatorIdentifier &_opid,
                           int64_t axis_,
                           const Op::Settings &settings_)
    : ElementWiseUnaryOp(_opid, settings_), axis(axis_) {}

std::unique_ptr<Op> LogSoftmaxOp::clone() const {
  return std::make_unique<LogSoftmaxOp>(*this);
}

std::vector<std::unique_ptr<Op>> LogSoftmaxOp::getGradOps() {
  throw error("LogSoftmaxOp should be removed by pattern 'LogSoftmaxOp' before "
              "call to getGradOps");
}

namespace {

static OpDefinition::DataTypes T = {DataType::FLOAT16, DataType::FLOAT};

static OpDefinition logSoftmaxOpDef({OpDefinition::Inputs({{"input", T}}),
                                     OpDefinition::Outputs({{"output", T}}),
                                     OpDefinition::Attributes({
                                         {"axis", {"*"}},
                                     })});

static OpCreator<LogSoftmaxOp> logSoftmaxOpCreator(
    OpDefinitions({{Onnx::Operators::LogSoftmax_1, logSoftmaxOpDef},
                   {Onnx::Operators::LogSoftmax_11, logSoftmaxOpDef}}),
    [](const OperatorIdentifier &_opid,
       const Op::Settings &settings,
       const Attributes &attr) -> std::unique_ptr<Op> {
      int64_t axis = attr.getAttribute<Attributes::Int>("axis", 1);

      return std::make_unique<LogSoftmaxOp>(_opid, axis, settings);
    },
    true);

} // namespace

} // namespace popart
