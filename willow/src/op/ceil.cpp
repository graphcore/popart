#include <memory>
#include <popart/op/ceil.hpp>
#include <popart/opmanager.hpp>

namespace popart {

std::vector<std::tuple<OperatorIdentifier, float>>
CeilOp::inplacePriorityDefault() const {
  // see T6768: choosing default inplace priorities
  return {{Onnx::CustomOperators::CeilInplace, 10}};
}

CeilInplaceOp::CeilInplaceOp(const CeilOp &ceil_op)
    : ElementWiseInplaceUnaryOp(Onnx::CustomOperators::CeilInplace,
                                ceil_op.getSettings()) {}

std::unique_ptr<Op>
CeilOp::getInplaceVariant(const OperatorIdentifier &operator_id) const {
  if (operator_id == Onnx::CustomOperators::CeilInplace) {
    return std::make_unique<CeilInplaceOp>(*this);
  }
  // catch remaining cases and throw an error
  return Op::getInplaceVariant(operator_id);
}

CeilOp::CeilOp(const OperatorIdentifier &_opid, const Op::Settings &settings_)
    : ElementWiseUnaryOp(_opid, settings_) {}

std::unique_ptr<Op> CeilOp::clone() const {
  return std::make_unique<CeilOp>(*this);
}

std::vector<std::unique_ptr<Op>> CeilOp::getGradOps() {
  throw error("PopART does not have a valid grad op corresponding to CeilOp");
}

std::unique_ptr<Op> CeilInplaceOp::clone() const {
  return std::make_unique<CeilInplaceOp>(*this);
}

namespace {

static OpDefinition::DataTypes T = {DataType::FLOAT16, DataType::FLOAT};

static OpDefinition ceilOpDef({OpDefinition::Inputs({{"X", T}}),
                               OpDefinition::Outputs({{"Y", T}}),
                               OpDefinition::Attributes({})});

static OpCreator<CeilOp>
    ceilOpCreator(OpDefinitions({{Onnx::Operators::Ceil_1, ceilOpDef},
                                 {Onnx::Operators::Ceil_6, ceilOpDef}}));

} // namespace
} // namespace popart
