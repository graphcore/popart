// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <memory>
#include <popart/op/getrandomseed.hpp>

#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/region.hpp"
#include "popart/tensorinfo.hpp"

namespace popart {
struct OperatorIdentifier;

GetRandomSeedOp::GetRandomSeedOp(const OperatorIdentifier &_opid,
                                 const Op::Settings &settings_)
    : Op(_opid, settings_) {}

void GetRandomSeedOp::setup() {
  outInfo(getUpdatedSeedOutIndex()) = inInfo(getSeedInIndex());
}

std::unique_ptr<Op> GetRandomSeedOp::clone() const {
  return std::make_unique<GetRandomSeedOp>(*this);
}

view::Regions GetRandomSeedOp::aliases(InIndex inIndex, OutIndex) const {
  return {view::Region::getFull(inShape(inIndex))};
}

// Modifies is the same as aliases
view::Regions GetRandomSeedOp::modifies(InIndex inIndex) const {
  return aliases(inIndex, 0);
}

namespace {} // namespace

} // namespace popart
