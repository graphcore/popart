// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <memory>
#include <string>
#include <popart/op/copyvarupdate.hpp>
#include <popart/opmanager.hpp>

#include "popart/datatype.hpp"
#include "popart/graphcoreoperators.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/op/varupdate.hpp"
#include "popart/region.hpp"

namespace popart {
struct OperatorIdentifier;

CopyVarUpdateOp::CopyVarUpdateOp(const Op::Settings &settings_)
    : VarUpdateWithUpdaterOp(Onnx::CustomOperators::CopyVarUpdate, settings_) {}

CopyVarUpdateOp::CopyVarUpdateOp(const OperatorIdentifier &opid_,
                                 const Op::Settings &settings_)
    : VarUpdateWithUpdaterOp(opid_, settings_) {}

std::unique_ptr<Op> CopyVarUpdateOp::clone() const {
  return std::make_unique<CopyVarUpdateOp>(*this);
}

view::Regions CopyVarUpdateOp::modifies(InIndex index) const {
  if (index == getVarToUpdateInIndex()) {
    // Modifies differs from base class since copy will
    // overwrite the tensor to update completely
    return {view::Region::getFull(inShape(index), view::AccessType::Write)};
  } else {
    return {view::Region::getEmpty(inRank(index))};
  }
}

namespace {
static OpDefinition::DataTypes T = {DataType::UINT32,
                                    DataType::INT32,
                                    DataType::FLOAT16,
                                    DataType::FLOAT};

static OpDefinition
    copyVarUpdateOpDef({OpDefinition::Inputs({{"variable", T}, {"updater", T}}),
                        OpDefinition::Outputs({{"alias", T}}),
                        OpDefinition::Attributes({})});

static OpCreator<CopyVarUpdateOp> copyVarUpdateOpCreator(OpDefinitions(
    {{Onnx::CustomOperators::CopyVarUpdate, copyVarUpdateOpDef}}));

} // namespace

} // namespace popart
