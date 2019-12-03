#include <memory>
#include <vector>
#include <popart/op/equal.hpp>
#include <popart/opmanager.hpp>
#include <popart/tensor.hpp>

namespace popart {

EqualOp::EqualOp(const OperatorIdentifier &_opid, const Op::Settings &settings_)
    : BinaryComparisonOp(_opid, settings_) {}

std::unique_ptr<Op> EqualOp::clone() const {
  return std::make_unique<EqualOp>(*this);
}

std::vector<std::unique_ptr<Op>> EqualOp::getGradOps() {
  throw error("PopART does not have a valid grad op corresponding to EqualOp");
}

namespace {

static OpDefinition::DataTypes T  = {DataType::UINT8,
                                    DataType::UINT16,
                                    DataType::UINT32,
                                    DataType::UINT64,
                                    DataType::INT8,
                                    DataType::INT16,
                                    DataType::INT32,
                                    DataType::INT64,
                                    DataType::FLOAT16,
                                    DataType::FLOAT,
                                    DataType::BOOL};
static OpDefinition::DataTypes T1 = {DataType::BOOL};

static OpDefinition equalOpDef({OpDefinition::Inputs({
                                    {"A", T},
                                    {"B", T},
                                }),
                                OpDefinition::Outputs({{"C", T1}}),
                                OpDefinition::Attributes({})});

static OpCreator<EqualOp>
    EqualOpCreator(OpDefinitions({{Onnx::Operators::Equal_1, equalOpDef},
                                  {Onnx::Operators::Equal_7, equalOpDef},
                                  {Onnx::Operators::Equal_11, equalOpDef}}));

} // namespace

} // namespace popart
