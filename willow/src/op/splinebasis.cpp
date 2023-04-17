// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#include <cmath>
#include <cstdint>
#include <vector>

#include "popart/error.hpp"
#include "popart/graphcoreoperators.hpp"
#include "popart/op/splinebasis.hpp"
#include "popart/opmanager.hpp"
#include "popart/opserialiser.hpp"

namespace popart {

SplineBasisOp::SplineBasisOp(const OperatorIdentifier &opid,
                             int degree,
                             const Op::Settings &settings)
    : Op(opid, settings), degree_(degree) {}

std::unique_ptr<Op> SplineBasisOp::clone() const {
  return std::make_unique<SplineBasisOp>(*this);
}

float SplineBasisOp::getSubgraphValue() const { return getLowSubgraphValue(); }

void SplineBasisOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("degree", degree_);
}

void SplineBasisOp::setup() {

  const auto pseudoInfo       = inInfo(pseudoIndex());
  const auto kernelSizeInfo   = inInfo(kernelSizeIndex());
  const auto isOpenSplineInfo = inInfo(isOpenSplineIndex());

  if (pseudoInfo.rank() != 2) {
    throw error("SplineBasisOp::setup: pseudo tensor is not a 2-D tensor. "
                "Pseudo tensor rank = {}.",
                pseudoInfo.rank());
  }
  if (kernelSizeInfo.rank() != 1) {
    throw error("SplineBasisOp::setup: kernelSize tensor is not a 1-D tensor. "
                "KernelSize tensor rank = {}.",
                kernelSizeInfo.rank());
  }
  if (isOpenSplineInfo.rank() != 1) {
    throw error(
        "SplineBasisOp::setup: isOpenSpline tensor is not a 1-D tensor. "
        "isOpenSpline tensor rank = {}.",
        isOpenSplineInfo.rank());
  }

  const auto E = pseudoInfo.dim(0);
  const auto D = pseudoInfo.dim(1);
  const auto S = static_cast<int64_t>(std::pow(getDegree() + 1, D) + 0.5);

  outInfo(outBasisIndex())       = {pseudoInfo.data_type(), Shape({E, S})};
  outInfo(outWeightIndexIndex()) = {kernelSizeInfo.data_type(), Shape({E, S})};
}

unsigned SplineBasisOp::getDegree() const noexcept { return degree_; }

namespace {

static OpDefinition::DataTypes Tf = {DataType::FLOAT16, DataType::FLOAT};

static OpDefinition::DataTypes Ti = {DataType::INT32};
static OpDefinition::DataTypes Tc = {DataType::UINT8};

static OpDefinition splinebasisOpDef(
    {OpDefinition::Inputs({
         {"pseudo", Tf},
         {"kernel_size", Ti},
         {"is_open_spline", Tc},
     }),
     OpDefinition::Outputs({{"basis", Tf}, {"weight_index", Ti}}),
     OpDefinition::Attributes({{"degree", {"[0-3]"}}})});

static constexpr bool isPublic = true;

static OpCreator<SplineBasisOp> SplineBasisOpCreator(
    OpDefinitions({
        {Onnx::CustomOperators::SplineBasis, splinebasisOpDef},
    }),
    [](const OpCreatorInfo &info) {
      const int64_t degree =
          info.attributes.getAttribute<Attributes::Int>("degree", 0);

      if (degree < 0 || degree > 3) {
        throw error("SplineBasisOp degree = {} is not valid: "
                    "must be between 0 and 3.",
                    degree);
      }

      return std::unique_ptr<Op>(
          new SplineBasisOp(info.opid, degree, info.settings));
    },
    isPublic);
} // namespace

} // namespace popart
