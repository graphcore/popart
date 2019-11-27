#include <memory>
#include <popart/graph.hpp>
#include <popart/op/atan.hpp>
#include <popart/opmanager.hpp>
#include <popart/region.hpp>
#include <popart/tensor.hpp>
#include <popart/util.hpp>

namespace popart {

std::vector<std::tuple<OperatorIdentifier, float>>
AtanOp::inplacePriorityDefault() const {
  // see T6768: choosing default inplace priorities
  return {{Onnx::CustomOperators::AtanInplace, 10}};
}

std::unique_ptr<Op>
AtanOp::getInplaceVariant(const OperatorIdentifier &operator_id) const {
  if (operator_id == Onnx::CustomOperators::AtanInplace) {
    return std::make_unique<AtanInplaceOp>(*this);
  }
  // catch remaining cases and throw an error
  return Op::getInplaceVariant(operator_id);
}

AtanInplaceOp::AtanInplaceOp(const AtanOp &atan_op)
    : ElementWiseInplaceUnaryOp(Onnx::CustomOperators::AtanInplace,
                                atan_op.getSettings()) {}

std::unique_ptr<Op> AtanInplaceOp::clone() const {
  return std::make_unique<AtanInplaceOp>(*this);
}

std::unique_ptr<Op> AtanOp::clone() const {
  return std::make_unique<AtanOp>(*this);
}

AtanOp::AtanOp(const OperatorIdentifier &_opid, const Op::Settings &settings_)
    : ElementWiseUnaryOp(_opid, settings_) {}

std::vector<std::unique_ptr<Op>> AtanOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(std::make_unique<AtanGradOp>(*this));
  return upops;
}

std::unique_ptr<Op> AtanGradOp::clone() const {
  return std::make_unique<AtanGradOp>(*this);
}

AtanGradOp::AtanGradOp(const AtanOp &fwdop)
    : ElementWiseNonLinearUnaryGradOp(Onnx::GradOperators::AtanGrad, fwdop) {}

namespace {

static OpDefinition::DataTypes T = {DataType::FLOAT16, DataType::FLOAT};

static OpDefinition atanOpDef({OpDefinition::Inputs({{"input", T}}),
                               OpDefinition::Outputs({{"output", T}}),
                               OpDefinition::Attributes({})});

static OpCreator<AtanOp> atanOpCreator(OpDefinitions({
    {Onnx::Operators::Atan_7, atanOpDef},
}));
} // namespace

} // namespace popart
