// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <memory>
#include <popart/op/mod.hpp>
#include <popart/opmanager.hpp>
#include <popart/tensor.hpp>

namespace popart {

ModOp::ModOp(const OperatorIdentifier &opId, const Op::Settings &settings)
    : ElementWiseBinaryOp(opId, settings) {}

std::unique_ptr<Op> ModOp::clone() const {
  return std::make_unique<ModOp>(*this);
}

std::vector<std::unique_ptr<Op>> ModOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;

  const auto &shapeIn0    = inShape(getArg0InIndex());
  const auto &shapeOutput = outShape(getOutIndex());

  upops.emplace_back(std::make_unique<ModArg0GradOp>(
      *this, npReductionAxis(shapeIn0, shapeOutput)));
  return upops;
}

ModArg0GradOp::ModArg0GradOp(const ModOp &op,
                             const std::vector<int64_t> &reductionAxes)
    : ElementWiseBinaryArg0GradOp(Onnx::GradOperators::ModArg0Grad,
                                  reductionAxes,
                                  op.inInfo(ModOp::getArg0InIndex()),
                                  op.getSettings()) {}

namespace {

static OpDefinition::DataTypes T = {DataType::UINT8,
                                    DataType::UINT16,
                                    DataType::UINT32,
                                    DataType::INT8,
                                    DataType::INT16,
                                    DataType::INT32,
                                    DataType::FLOAT16,
                                    DataType::FLOAT,
                                    DataType::DOUBLE};

static OpDefinition modOpDef({OpDefinition::Inputs({
                                  {"A", T},
                                  {"B", T},
                              }),
                              OpDefinition::Outputs({{"C", T}}),
                              OpDefinition::Attributes({{"fmod", {"*"}}})});

static OpCreator<ModOp>
    modOpCreator(OpDefinitions({{Onnx::Operators::Mod_10, modOpDef}}));
} // namespace

} // namespace popart
