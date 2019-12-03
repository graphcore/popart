#include <memory>
#include <popart/op/floor.hpp>
#include <popart/opmanager.hpp>

namespace popart {

std::vector<std::tuple<OperatorIdentifier, float>>
FloorOp::inplacePriorityDefault() const {
  // see T6768: choosing default inplace priorities
  return {{Onnx::CustomOperators::FloorInplace, 10}};
}

FloorInplaceOp::FloorInplaceOp(const FloorOp &floor_op)
    : ElementWiseInplaceUnaryOp(Onnx::CustomOperators::FloorInplace,
                                floor_op.getSettings()) {}

std::unique_ptr<Op>
FloorOp::getInplaceVariant(const OperatorIdentifier &operator_id) const {
  if (operator_id == Onnx::CustomOperators::FloorInplace) {
    return std::make_unique<FloorInplaceOp>(*this);
  }
  // catch remaining cases and throw an error
  return Op::getInplaceVariant(operator_id);
}

FloorOp::FloorOp(const OperatorIdentifier &_opid, const Op::Settings &settings_)
    : ElementWiseUnaryOp(_opid, settings_) {}

std::unique_ptr<Op> FloorOp::clone() const {
  return std::make_unique<FloorOp>(*this);
}

std::vector<std::unique_ptr<Op>> FloorOp::getGradOps() {
  throw error("PopART does not have a valid grad op corresponding to FloorOp");
}

std::unique_ptr<Op> FloorInplaceOp::clone() const {
  return std::make_unique<FloorInplaceOp>(*this);
}

namespace {

static OpDefinition::DataTypes T = {DataType::FLOAT16, DataType::FLOAT};

static OpDefinition floorOpDef({OpDefinition::Inputs({{"X", T}}),
                                OpDefinition::Outputs({{"Y", T}}),
                                OpDefinition::Attributes({})});

static OpCreator<FloorOp>
    floorOpCreator(OpDefinitions({{Onnx::Operators::Floor_1, floorOpDef},
                                  {Onnx::Operators::Floor_6, floorOpDef}}));

} // namespace
} // namespace popart
