// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <memory>
#include <vector>
#include <popart/op/modifyrandomseed.hpp>
#include <popart/tensor.hpp>

#include "popart/datatype.hpp"
#include "popart/error.hpp"
#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/tensorindex.hpp"
#include "popart/tensorinfo.hpp"
#include "popart/tensorlocation.hpp"

namespace popart {

struct OperatorIdentifier;

ModifyRandomSeedOp::ModifyRandomSeedOp(const OperatorIdentifier &_opid,
                                       const Op::Settings &settings_)
    : Op(_opid, settings_) {}

void ModifyRandomSeedOp::setup() {

  auto seedInInfo     = inInfo(getSeedInIndex());
  auto modifierInInfo = inInfo(ModifyRandomSeedOp::getSeedModifierInIndex());

  // First param is the seed, expecting two UINT32s.
  if ((seedInInfo.dataType() != DataType::UINT32) ||
      (seedInInfo.shape() != Shape({2}))) {
    throw error("Expected {} to be a tensor of type uint32 [2] (found {} {})",
                input->tensor(getSeedInIndex())->id,
                seedInInfo.dataType(),
                seedInInfo.shape());
  }

  // Second param a modifying tensor, expecting one UINT32.
  if ((modifierInInfo.dataType() != DataType::UINT32) ||
      (modifierInInfo.shape() != Shape({}))) {
    throw error("Expected {} to be a tensor of type uint32 [] (found {} {})",
                input->tensor(ModifyRandomSeedOp::getSeedModifierInIndex())->id,
                modifierInInfo.dataType(),
                modifierInInfo.shape());
  }

  outInfo(getModifiedSeedOutIndex()) = seedInInfo;
}

std::unique_ptr<Op> ModifyRandomSeedOp::clone() const {
  return std::make_unique<ModifyRandomSeedOp>(*this);
}

} // namespace popart
