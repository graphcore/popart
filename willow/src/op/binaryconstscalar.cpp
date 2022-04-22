// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <map>
#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <vector>
#include <popart/op/binaryconstscalar.hpp>
#include <popart/opmanager.hpp>

#include "popart/attributes.hpp"
#include "popart/datatype.hpp"
#include "popart/error.hpp"
#include "popart/graphcoreoperators.hpp"
#include "popart/logging.hpp"
#include "popart/op.hpp"

namespace popart {

std::vector<std::unique_ptr<Op>> BinaryConstScalarOp::getGradOps() {
  throw error(
      "BinaryConstScalar Op should be removed by Pattern before auto-grad");
}

std::unique_ptr<Op> BinaryConstScalarOp::clone() const {
  return std::make_unique<BinaryConstScalarOp>(*this);
}

namespace {

auto fromString(const std::string &n) {
  const static std::map<std::string, BinaryConstScalarOp::Type> M{
      {"Mul", BinaryConstScalarOp::Type::Mul},
      {"Div", BinaryConstScalarOp::Type::Div},
      {"Add", BinaryConstScalarOp::Type::Add},
      {"Sub", BinaryConstScalarOp::Type::Sub},
      {"Pow", BinaryConstScalarOp::Type::Pow}};

  const auto found = M.find(n);
  if (found == M.cend()) {
    std::ostringstream oss;
    oss << "Unrecognised opType " << n << ", expected one of ( ";
    for (const auto &x : M) {
      oss << x.first;
      oss << ' ';
    }
    oss << ')';
  }
  return found->second;
} // namespace

// I've just copied these from mul.cpp:
static OpDefinition::DataTypes T = {DataType::UINT32,
                                    DataType::UINT64,
                                    DataType::INT32,
                                    DataType::INT64,
                                    DataType::FLOAT16,
                                    DataType::FLOAT};

static OpDefinition binaryConstScalarOpDef(
    {OpDefinition::Inputs({{"X", T}}),
     OpDefinition::Outputs({{"Y", T}}),

     // I'm not sure what the * means.
     OpDefinition::Attributes(
         {{"value", {"*"}}, {"op", {"*"}}, {"scalar_in_index", {"*"}}})});

static OpCreator<BinaryConstScalarOp> creator(
    OpDefinitions({{Onnx::CustomOperators::BinaryConstScalar,
                    binaryConstScalarOpDef}}),
    [](const OpCreatorInfo &info) {
      return std::unique_ptr<Op>(new BinaryConstScalarOp(
          info.opid,
          info.attributes.getAttribute<Attributes::Float>("value"),
          fromString(info.attributes.getAttribute<Attributes::String>("op")),
          info.attributes.getAttribute<Attributes::Int>("scalar_in_index"),
          info.settings));
    },
    true);

} // namespace

} // namespace popart
