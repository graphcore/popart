#include <memory>
#include <popart/ir.hpp>
#include <popart/op/isnan.hpp>
#include <popart/opmanager.hpp>
#include <popart/tensor.hpp>

namespace popart {

IsNaN::IsNaN(const OperatorIdentifier &_opid, const Op::Settings &settings_)
    : ElementWiseUnaryBooleanOp(_opid, settings_) {}

std::unique_ptr<Op> IsNaN::clone() const {
  return std::make_unique<IsNaN>(*this);
}

OperatorIdentifier IsNaN::getOpId(const Ir &) {
  return Onnx::Operators::IsNaN_9;
}

namespace {

static OpDefinition::DataTypes T1 = {DataType::FLOAT, DataType::FLOAT16};
static OpDefinition::DataTypes T2 = {DataType::BOOL};

static OpDefinition isNanOpDef({OpDefinition::Inputs({{"x", T1}}),
                                OpDefinition::Outputs({{"y", T2}}),
                                OpDefinition::Attributes({})});

static OpCreator<IsNaN>
    IsNaNCreator(OpDefinitions({{Onnx::Operators::IsNaN_9, isNanOpDef}}));
} // namespace

} // namespace popart
