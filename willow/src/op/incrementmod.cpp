// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <memory>
#include <string>
#include <popart/op/incrementmod.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>

#include "popart/attributes.hpp"
#include "popart/datatype.hpp"
#include "popart/graphcoreoperators.hpp"
#include "popart/op.hpp"
#include "popart/op/elementwise.hpp"

namespace popart {
struct OperatorIdentifier;

IncrementModOp::IncrementModOp(const OperatorIdentifier &opId,
                               double increment_,
                               double modulus_,
                               const Op::Settings &settings)
    : ElementWiseUnaryOp(opId, settings), increment(increment_),
      modulus(modulus_) {}

std::unique_ptr<Op> IncrementModOp::clone() const {
  return std::make_unique<IncrementModOp>(*this);
}

void IncrementModOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("increment", increment);
  os.appendAttribute("modulus", modulus);
}

IncrementModInplaceOp::IncrementModInplaceOp(double increment_,
                                             double modulus_,
                                             const Op::Settings &settings)
    : ElementWiseInplaceUnaryOp(Onnx::CustomOperators::IncrementModInplace_1,
                                settings),
      increment(increment_), modulus(modulus_) {}

IncrementModInplaceOp::IncrementModInplaceOp(
    const IncrementModOp &incrementModOp)
    : ElementWiseInplaceUnaryOp(Onnx::CustomOperators::IncrementModInplace_1,
                                incrementModOp.getSettings()),
      increment(incrementModOp.getIncrement()),
      modulus(incrementModOp.getModulus()) {}

std::unique_ptr<Op> IncrementModInplaceOp::clone() const {
  return std::make_unique<IncrementModInplaceOp>(*this);
}

void IncrementModInplaceOp::appendOutlineAttributes(
    OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("increment", increment);
  os.appendAttribute("modulus", modulus);
}

namespace {

static OpDefinition::DataTypes T = {DataType::UINT32,
                                    DataType::INT32,
                                    DataType::FLOAT16,
                                    DataType::FLOAT};

static OpDefinition incrementModOpDef({OpDefinition::Inputs({
                                           {"X", T},
                                       }),
                                       OpDefinition::Outputs({{"Y", T}}),
                                       {}});

static OpCreator<IncrementModOp> incrementModOpCreator(
    OpDefinitions({{Onnx::CustomOperators::IncrementMod_1, incrementModOpDef}}),
    [](const OpCreatorInfo &info) {
      double increment =
          info.attributes.getAttribute<Attributes::Float>("increment");
      double modulus =
          info.attributes.getAttribute<Attributes::Float>("modulus");
      return std::unique_ptr<IncrementModOp>(
          new IncrementModOp(info.opid, increment, modulus, info.settings));
    },
    true);

} // namespace

} // namespace popart
