#include <algorithm>
#include <vector>

#include <memory>
#include <popart/op/argmax.hpp>
#include <popart/opmanager.hpp>
#include <popart/tensor.hpp>

namespace popart {

std::unique_ptr<Op> ArgMaxOp::clone() const {
  return std::make_unique<ArgMaxOp>(*this);
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
static OpDefinition::DataTypes T1 = {DataType::INT64};

static OpDefinition argMaxOpDef(
    {OpDefinition::Inputs({{"data", T}}),
     OpDefinition::Outputs({{"reduced", T1}}),
     OpDefinition::Attributes({{"axis", {"*"}}, {"keepdims", {"0..1"}}})});

std::unique_ptr<Op> argMaxFactory(const OperatorIdentifier &_opid,
                                  const Op::Settings &settings,
                                  const Attributes &attr) {
  int64_t axis     = attr.getAttribute<Attributes::Int>("axis", 0);
  int64_t keepdims = attr.getAttribute<Attributes::Int>("keepdims", 1);

  return std::make_unique<ArgMaxOp>(_opid, axis, keepdims, settings);
}

static OpCreator<ArgMaxOp>
    ArgMaxOpCreator(OpDefinitions({{Onnx::Operators::ArgMax_1, argMaxOpDef},
                                   {Onnx::Operators::ArgMax_11, argMaxOpDef}}),
                    argMaxFactory,
                    true);
} // namespace

} // namespace popart
