// Copyright (c) 2023 Graphcore Ltd. All rights reserved.
#include <vector>

#include "popart/error.hpp"
#include "popart/graphcoreoperators.hpp"
#include "popart/op/splineweighting.hpp"
#include "popart/opmanager.hpp"
#include "popart/opserialiser.hpp"

namespace popart {

SplineWeightingOp::SplineWeightingOp(const OperatorIdentifier &opid,
                                     const Op::Settings &settings)
    : Op(opid, settings) {}

std::unique_ptr<Op> SplineWeightingOp::clone() const {
  return std::make_unique<SplineWeightingOp>(*this);
}

float SplineWeightingOp::getSubgraphValue() const {
  return getLowSubgraphValue();
}

void SplineWeightingOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
}

void SplineWeightingOp::setup() {

  const auto inputInfo       = inInfo(inputIndex());
  const auto weightInfo      = inInfo(weightIndex());
  const auto basisInfo       = inInfo(basisIndex());
  const auto weightIndexInfo = inInfo(weightIndexIndex());

  if (inputInfo.rank() != 2) {
    throw error("SplineWeightingOp::setup: input tensor is not a 2-D tensor. "
                "Input tensor rank = {}.",
                inputInfo.rank());
  }
  if (weightInfo.rank() != 3) {
    throw error("SplineWeightingOp::setup: weight tensor is not a 3-D tensor. "
                "Weight tensor rank = {}.",
                weightInfo.rank());
  }
  if (basisInfo.rank() != 2) {
    throw error("SplineWeightingOp::setup: basis tensor is not a 2-D tensor. "
                "Basis tensor rank = {}.",
                basisInfo.rank());
  }
  if (weightIndexInfo.rank() != 2) {
    throw error(
        "SplineWeightingOp::setup: weightIndex tensor is not a 2-D tensor. "
        "weightIndex tensor rank = {}.",
        weightIndexInfo.rank());
  }

  if (inputInfo.dataType() != weightInfo.dataType()) {
    throw error("SplineWeightingOp::setup: The input and weight tensors have "
                "different data types. The input tensor data type = {}. The "
                "weight tensor data type = {}.",
                inputInfo.data_type_lcase(),
                weightInfo.data_type_lcase());
  }

  if (inputInfo.dataType() != basisInfo.dataType()) {
    throw error("SplineWeightingOp::setup: The input and basis tensors have "
                "different data types. The input tensor data type = {}. The "
                "basis tensor data type = {}.",
                inputInfo.data_type_lcase(),
                basisInfo.data_type_lcase());
  }

  const auto E     = inputInfo.dim(0);
  const auto M_out = weightInfo.dim(2);

  outInfo(outputIndex()) = {inputInfo.data_type(), Shape({E, M_out})};
}

namespace {

static OpDefinition::DataTypes Tf = {DataType::FLOAT16, DataType::FLOAT};

static OpDefinition::DataTypes Ti = {DataType::INT32};

static OpDefinition
    splineweightingOpDef({OpDefinition::Inputs({
                              {"input", Tf},
                              {"weight", Tf},
                              {"basis", Tf},
                              {"weight_index", Ti},
                          }),
                          OpDefinition::Outputs({{"output", Tf}}),
                          OpDefinition::Attributes({})});

static constexpr bool isPublic = true;

static OpCreator<SplineWeightingOp> SplineWeightingOpCreator(
    OpDefinitions({
        {Onnx::CustomOperators::SplineWeighting, splineweightingOpDef},
    }),
    [](const OpCreatorInfo &info) {
      return std::unique_ptr<Op>(
          new SplineWeightingOp(info.opid, info.settings));
    },
    isPublic);
} // namespace

} // namespace popart
