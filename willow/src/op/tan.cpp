#include <memory>
#include <popart/op/tan.hpp>
#include <popart/opmanager.hpp>
#include <popart/tensor.hpp>

namespace popart {

TanOp::TanOp(const OperatorIdentifier &_opid, const Op::Settings &settings_)
    : ElementWiseUnaryOp(_opid, settings_) {}

std::unique_ptr<Op> TanOp::clone() const {
  return std::make_unique<TanOp>(*this);
}

std::vector<std::unique_ptr<Op>> TanOp::getGradOps() {
  throw error("TanOp should be removed by pattern 'TanOp' before call to "
              "TanOp::getGradOps");
}

namespace {

static OpDefinition::DataTypes T = {DataType::FLOAT16, DataType::FLOAT};

static OpDefinition tanOpDef({OpDefinition::Inputs({
                                  {"input", T},
                              }),
                              OpDefinition::Outputs({{"output", T}}),
                              OpDefinition::Attributes({})});

static OpCreator<TanOp> tanOpCreator(OpDefinitions({
    {Onnx::Operators::Tan_7, tanOpDef},
}));
} // namespace

} // namespace popart
