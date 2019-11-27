#include <memory>
#include <popart/op/argmin.hpp>
#include <popart/opmanager.hpp>
#include <popart/tensor.hpp>

namespace popart {

std::unique_ptr<Op> ArgMinOp::clone() const {
  return std::make_unique<ArgMinOp>(*this);
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

static OpDefinition argMinOpDef(
    {OpDefinition::Inputs({{"data", T}}),
     OpDefinition::Outputs({{"reduced", T1}}),
     OpDefinition::Attributes({{"axis", {"*"}}, {"keepdims", {"0..1"}}})});

std::unique_ptr<Op> argMinFactory(const OperatorIdentifier &_opid,
                                  const Op::Settings &settings,
                                  const Attributes &attr) {
  int64_t axis     = attr.getAttribute<Attributes::Int>("axis", 0);
  int64_t keepdims = attr.getAttribute<Attributes::Int>("keepdims", 1);

  return std::make_unique<ArgMinOp>(_opid, axis, keepdims, settings);
}

static OpCreator<ArgMinOp>
    argMinOpCreator(OpDefinitions({{Onnx::Operators::ArgMin_1, argMinOpDef},
                                   {Onnx::Operators::ArgMin_11, argMinOpDef}}),
                    argMinFactory,
                    true);
} // namespace

} // namespace popart
