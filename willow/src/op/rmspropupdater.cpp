// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <limits>
#include <memory>
#include <popart/ir.hpp>
#include <popart/op/rmspropupdater.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/region.hpp>
#include <popart/tensornames.hpp>

namespace popart {

RMSPropUpdaterOp::RMSPropUpdaterOp(OptimizerValue eps,
                                   bool TFVariant,
                                   const Op::Settings &opSettings)
    : Op(Onnx::CustomOperators::RMSPropUpdater, opSettings), initEps(eps),
      TFVariant(TFVariant) {}

void RMSPropUpdaterOp::setup() {
  outInfo(getUpdaterOutIndex()) = inInfo(getGradInIndex());
}

std::unique_ptr<Op> RMSPropUpdaterOp::clone() const {
  return std::make_unique<RMSPropUpdaterOp>(*this);
}

void RMSPropUpdaterOp::appendOutlineAttributes(OpSerialiserBase &os) const {

  Op::appendOutlineAttributes(os);

  if (initEps.isConst()) {
    os.appendAttribute("const eps", initEps.val());
  }
  os.appendAttribute("tf variant", static_cast<int>(TFVariant));
}

ReplicatedTensorShardingIndices
RMSPropUpdaterOp::getReplicatedTensorShardingIndices() const {
  std::set<InIndex> inIndices;

  inIndices.insert(RMSPropUpdaterOp::getGradInIndex());
  inIndices.insert(RMSPropUpdaterOp::getAccl1InIndex());

  if (hasInput(RMSPropUpdaterOp::getAccl2InIndex())) {
    inIndices.insert(RMSPropUpdaterOp::getAccl2InIndex());
  }

  return {{inIndices, {RMSPropUpdaterOp::getUpdaterOutIndex()}}};
}

} // namespace popart
