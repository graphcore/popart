// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <onnx/onnx_pb.h>
#include <onnxutil.hpp>
#include <string>
#include <vector>
#include <popart/op/cast.hpp>
#include <popart/opmanager.hpp>

#include "popart/attributes.hpp"
#include "popart/datatype.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/operators.hpp"
#include "popart/tensorinfo.hpp"

namespace popart {
struct OperatorIdentifier;

CastOp::CastOp(const OperatorIdentifier &_opid,
               DataType _to,
               const Op::Settings &settings_)
    : Op(_opid, settings_), to(_to) {}

std::unique_ptr<Op> CastOp::clone() const {
  return std::make_unique<CastOp>(*this);
}

std::vector<std::unique_ptr<Op>> CastOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  DataType fromDataType = inInfo(getInIndex()).dataType();

  // Only insert into graph if source is a floating point.
  if (fromDataType == DataType::FLOAT || fromDataType == DataType::FLOAT16) {
    upops.emplace_back(std::make_unique<CastGradOp>(*this));
  }
  return upops;
}

void CastOp::setup() {
  auto info = inInfo(getInIndex());
  // Change data type
  info.set(to);
  outInfo(getOutIndex()) = info;
}

bool CastOp::canBeReplacedByIdentity() const {
  if (!hasInput(getInIndex())) {
    // Cannot determine whether Op can be replaced by identity, as its input
    // is not connected. Return default false.
    return false;
  }

  return toDataType() == inInfo(getInIndex()).dataType();
}

CastGradOp::CastGradOp(const CastOp &fwdOp)
    : CastOp(Onnx::GradOperators::CastGrad,
             fwdOp.inInfo(getInIndex()).dataType(),
             fwdOp.getSettings()) {}

std::unique_ptr<Op> CastGradOp::clone() const {
  return std::make_unique<CastGradOp>(*this);
}

const std::vector<GradInOutMapper> &CastGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getInIndex(), CastOp::getOutIndex(), GradOpInType::GradOut}};

  return inInfo;
}

const std::map<int, int> &CastGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {getOutIndex(), CastOp::getInIndex()}};

  return outInfo;
}

namespace {

static OpDefinition::DataTypes T1 = {DataType::FLOAT16,
                                     DataType::FLOAT,
                                     DataType::INT8,
                                     DataType::INT16,
                                     DataType::INT32,
                                     DataType::UINT8,
                                     DataType::UINT16,
                                     DataType::UINT32,
                                     DataType::BOOL};
static OpDefinition::DataTypes T2 = {DataType::FLOAT16,
                                     DataType::FLOAT,
                                     DataType::INT8,
                                     DataType::INT16,
                                     DataType::INT32,
                                     DataType::UINT8,
                                     DataType::UINT16,
                                     DataType::UINT32,
                                     DataType::BOOL};

static OpDefinition castOpDef(
    {OpDefinition::Inputs({{"input", T1}}),
     OpDefinition::Outputs({{"output", T2}}),
     OpDefinition::Attributes(
         {{"to",
           {"FLOAT|FLOAT16|INT8|INT16|INT32|UINT8|UINT16|UINT32|BOOL"}}})});

static OpCreator<CastOp> castOpCreator(
    OpDefinitions({
        {Onnx::Operators::Cast_6, castOpDef},
        {Onnx::Operators::Cast_9, castOpDef},
    }),
    [](const OpCreatorInfo &info) {
      int64_t i64_to;
      info.attributes.set(i64_to, "to");
      auto tpdt_to = static_cast<ONNX_NAMESPACE::TensorProto_DataType>(i64_to);
      DataType dt_to = onnxutil::getDataType(tpdt_to);

      return std::make_unique<CastOp>(info.opid, dt_to, info.settings);
    },
    true);
} // namespace

} // namespace popart
