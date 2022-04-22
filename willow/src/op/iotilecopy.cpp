// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <map>
#include <memory>
#include <set>
#include <utility>
#include <popart/op/iotilecopy.hpp>
#include <popart/tensorindex.hpp>

#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/tensorinfo.hpp"
#include "popart/tensorlocation.hpp"

namespace popart {
struct OperatorIdentifier;

IoTileCopyOp::IoTileCopyOp(const OperatorIdentifier &_opid,
                           const Op::Settings &settings_)
    : Op(_opid, settings_) {}

std::unique_ptr<Op> IoTileCopyOp::clone() const {
  return std::make_unique<IoTileCopyOp>(*this);
}

void IoTileCopyOp::setup() {
  for (auto &idx_tensor : input->tensorMap()) {
    auto idx     = idx_tensor.first;
    outInfo(idx) = inInfo(idx);
  }
}

VGraphIdAndTileSet
IoTileCopyOp::getIntrospectionInVirtualGraphId(InIndex,
                                               std::set<OpId> &visited) const {
  return {hasVirtualGraphId() ? getVirtualGraphId() : unusedVGraphId,
          settings.tileSet == TileSet::Compute ? TileSet::IO
                                               : TileSet::Compute};
}

VGraphIdAndTileSet
IoTileCopyOp::getIntrospectionOutVirtualGraphId(OutIndex,
                                                std::set<OpId> &visited) const {
  return {hasVirtualGraphId() ? getVirtualGraphId() : unusedVGraphId,
          settings.tileSet};
}

} // namespace popart
