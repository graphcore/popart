#include <memory>
#include <vector>
#include <popart/op/greater.hpp>
#include <popart/opmanager.hpp>
#include <popart/tensor.hpp>

namespace popart {

GreaterOp::GreaterOp(const OperatorIdentifier &_opid,
                     const Op::Settings &settings_)
    : BinaryComparisonOp(_opid, settings_) {}

std::unique_ptr<Op> GreaterOp::clone() const {
  return std::make_unique<GreaterOp>(*this);
}

std::vector<std::unique_ptr<Op>> GreaterOp::getGradOps() {
  throw error(
      "PopART does not have a valid grad op corresponding to GreaterOp");
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
                                    DataType::FLOAT};
static OpDefinition::DataTypes T1 = {DataType::BOOL};

static OpDefinition greaterOpDef({OpDefinition::Inputs({
                                      {"A", T},
                                      {"B", T},
                                  }),
                                  OpDefinition::Outputs({{"C", T1}}),
                                  OpDefinition::Attributes({})});

// Note : Don't support attributes axis or broadcast for version 1

static OpCreator<GreaterOp> GreaterOpCreator(
    OpDefinitions({{Onnx::Operators::Greater_1, greaterOpDef},
                   {Onnx::Operators::Greater_7, greaterOpDef},
                   {Onnx::Operators::Greater_9, greaterOpDef}}));
} // namespace

} // namespace popart
