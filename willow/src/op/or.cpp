#include <memory>
#include <vector>
#include <popart/op/or.hpp>
#include <popart/opmanager.hpp>
#include <popart/tensor.hpp>

namespace popart {

OrOp::OrOp(const OperatorIdentifier &_opid, const Op::Settings &settings_)
    : BinaryComparisonOp(_opid, settings_) {}

std::unique_ptr<Op> OrOp::clone() const {
  return std::make_unique<OrOp>(*this);
}

std::vector<std::unique_ptr<Op>> OrOp::getGradOps() {
  throw error("PopART does not have a valid grad op corresponding to OrOp");
}

namespace {

static OpDefinition::DataTypes T  = {DataType::BOOL};
static OpDefinition::DataTypes T1 = {DataType::BOOL};

static OpDefinition orOpDef({OpDefinition::Inputs({
                                 {"A", T},
                                 {"B", T},
                             }),
                             OpDefinition::Outputs({{"C", T1}}),
                             OpDefinition::Attributes({})});

static OpCreator<OrOp> OrOpCreator(OpDefinitions(
    {{Onnx::Operators::Or_1, orOpDef}, {Onnx::Operators::Or_7, orOpDef}}));
} // namespace

} // namespace popart
