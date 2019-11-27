#include <memory>
#include <popart/op/square.hpp>
#include <popart/opmanager.hpp>
#include <popart/tensor.hpp>

namespace popart {

SquareOp::SquareOp(const OperatorIdentifier &_opid,
                   const Op::Settings &settings_)
    : ElementWiseUnaryOp(_opid, settings_) {}

std::unique_ptr<Op> SquareOp::clone() const {
  return std::unique_ptr<Op>(new SquareOp(*this));
}

std::vector<std::unique_ptr<Op>> SquareOp::getGradOps() {
  throw error("Grad op has not been implemented for SquareOp");
}

namespace {

static OpDefinition::DataTypes T = {DataType::FLOAT16, DataType::FLOAT};

static OpDefinition squareOpDef({OpDefinition::Inputs({{"X", T}}),
                                 OpDefinition::Outputs({{"Y", T}}),
                                 OpDefinition::Attributes({})});

// There is no defs.cc for this operation
static OpCreator<SquareOp> squareOpCreator(OpDefinitions({
    {Onnx::CustomOperators::Square, squareOpDef},
}));
} // namespace

} // namespace popart
