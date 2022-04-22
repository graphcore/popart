// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <popart/op/fmod.hpp>
#include <popart/opmanager.hpp>

#include "popart/datatype.hpp"
#include "popart/graphcoreoperators.hpp"
#include "popart/op.hpp"
#include "popart/op/elementwise.hpp"
#include "popart/tensorinfo.hpp"

namespace popart {
struct OperatorIdentifier;

FmodOp::FmodOp(const OperatorIdentifier &opId, const Op::Settings &settings)
    : ElementWiseBinaryOp(opId, settings) {}

std::unique_ptr<Op> FmodOp::clone() const {
  return std::make_unique<FmodOp>(*this);
}

std::vector<std::unique_ptr<Op>> FmodOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;

  const auto &shapeIn0    = inShape(getArg0InIndex());
  const auto &shapeOutput = outShape(getOutIndex());

  upops.emplace_back(std::make_unique<FmodArg0GradOp>(
      *this, npReductionAxis(shapeIn0, shapeOutput)));
  return upops;
}

FmodArg0GradOp::FmodArg0GradOp(const FmodOp &op,
                               const std::vector<int64_t> &reductionAxes)
    : ElementWiseBinaryArg0GradOp(Onnx::GradOperators::FmodArg0Grad,
                                  reductionAxes,
                                  op.inInfo(FmodOp::getArg0InIndex()),
                                  op.getSettings()) {}

std::unique_ptr<Op> FmodArg0GradOp::clone() const {
  return std::make_unique<FmodArg0GradOp>(*this);
}

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

static OpDefinition fmodOpDef({OpDefinition::Inputs({
                                   {"A", T},
                                   {"B", T},
                               }),
                               OpDefinition::Outputs({{"C", T}}),
                               {}});

static OpCreator<FmodOp> fmodOpCreator(
    OpDefinitions({{Onnx::AiGraphcore::OpSet1::Fmod, fmodOpDef}}));
} // namespace

} // namespace popart
