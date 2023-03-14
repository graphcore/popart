// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#include <vector>

#include "popart/error.hpp"
#include "popart/graphcoreoperators.hpp"
#include "popart/op/bucketize.hpp"
#include "popart/opmanager.hpp"
#include "popart/opserialiser.hpp"

namespace popart {

BucketizeOp::BucketizeOp(const OperatorIdentifier &opid,
                         bool right,
                         const Op::Settings &settings)
    : Op(opid, settings), right_(right) {}

std::unique_ptr<Op> BucketizeOp::clone() const {
  return std::make_unique<BucketizeOp>(*this);
}

float BucketizeOp::getSubgraphValue() const { return getLowSubgraphValue(); }

void BucketizeOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("right", right_);
}

void BucketizeOp::setup() {

  const auto boundariesInfo = inInfo(boundariesInIndex());
  const auto inputInfo      = inInfo(inIndex());
  const auto boundariesRank = boundariesInfo.rank();

  if (boundariesRank != 1) {
    throw error("BucketizeOp::setup The boundaries tensor is not a 1-D tensor. "
                "The boundaries tensor rank = {}.",
                boundariesRank);
  }

  if (inputInfo.dataType() != boundariesInfo.dataType()) {
    throw error("BucketizeOp::setup The input and boundaries tensors have "
                "different data types. The input tensor data type = {}. The "
                "boundaries tensor data type = {}.",
                inputInfo.data_type_lcase(),
                boundariesInfo.data_type_lcase());
  }

  outInfo(outIndex()) = {DataType::INT32, inputInfo.shape()};
}

bool BucketizeOp::isRight() const noexcept { return right_; }

namespace {

static OpDefinition::DataTypes T = {DataType::UINT32,
                                    DataType::INT32,
                                    DataType::FLOAT16,
                                    DataType::FLOAT};

static OpDefinition::DataTypes TOut = {DataType::INT32};

static OpDefinition
    bucketizeOpDef({OpDefinition::Inputs({
                        {"input", T},
                        {"boundaries", T}, // optional
                    }),
                    OpDefinition::Outputs({{"output", TOut}}),
                    OpDefinition::Attributes({{"right", {"[0-1]"}}})});

static constexpr bool isPublic = true;

static OpCreator<BucketizeOp> BucketizeOpCreator(
    OpDefinitions({
        {Onnx::CustomOperators::Bucketize, bucketizeOpDef},
    }),
    [](const OpCreatorInfo &info) {
      const int64_t right =
          info.attributes.getAttribute<Attributes::Int>("right", 0);

      if (right != 0 && right != 1) {
        throw error("BucketizeOp right = {} is not valid: "
                    "must be either 0 or 1.",
                    right);
      }

      return std::unique_ptr<Op>(
          new BucketizeOp(info.opid, right, info.settings));
    },
    isPublic);
} // namespace

} // namespace popart
