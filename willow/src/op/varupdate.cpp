// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <limits>
#include <memory>
#include <popart/ir.hpp>
#include <popart/logging.hpp>
#include <popart/op/varupdate.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/region.hpp>
#include <popart/tensornames.hpp>

namespace popart {
VarUpdateOp::VarUpdateOp(const OperatorIdentifier &_opid,
                         const Op::Settings &settings_)
    : Op(_opid, settings_) {
  // TODO: Remove with T19212
  if (getIr().getSessionOptions().executionPhaseSettings.phases < 2 &&
      getIr().getSessionOptions().batchSerializationSettings.factor < 2 &&
      getIr().getSessionOptions().delayVarUpdates) {
    settings.schedulePriority = std::numeric_limits<double>::lowest();
  }
}

VarUpdateWithUpdaterOp::VarUpdateWithUpdaterOp(const OperatorIdentifier &opid_,
                                               const Op::Settings &settings_)
    : VarUpdateOp(opid_, settings_) {}

void VarUpdateWithoutUpdaterOp::setup() {
  outInfo(getUpdatedVarOutIndex()) = inInfo(getVarToUpdateInIndex());
}

void VarUpdateWithUpdaterOp::setup() {
  auto info0 = inInfo(getVarToUpdateInIndex());
  auto info1 = inInfo(getUpdaterInIndex());
  if (info0 != info1) {
    std::ostringstream oss;
    oss << "In VarUpdateOp::setup(), the VarToUpdate has TensorInfo \n"
        << info0 << "\nbut the Updater has TensorInfo\n"
        << info1;

    // TODO T12001 : sort this out (serialize matmuls meets grad accl)
    logging::ir::warn(oss.str());
  }
  outInfo(getUpdatedVarOutIndex()) = info0;
}

view::Regions VarUpdateOp::aliases(InIndex in, OutIndex) const {
  if (in == getVarToUpdateInIndex()) {
    return {view::Region::getFull(inShape(in))};
  } else {
    return {view::Region::getEmpty(inRank(in))};
  }
}

view::Regions VarUpdateOp::modifies(InIndex index) const {
  return aliases(index, 0);
}

ReplicatedTensorShardingIndices
VarUpdateOp::getReplicatedTensorShardingIndices() const {
  return {{{VarUpdateOp::getVarToUpdateInIndex()},
           {VarUpdateOp::getUpdatedVarOutIndex()}}};
}

ReplicatedTensorShardingIndices
VarUpdateWithUpdaterOp::getReplicatedTensorShardingIndices() const {
  return {{{VarUpdateWithUpdaterOp::getVarToUpdateInIndex(),
            VarUpdateWithUpdaterOp::getUpdaterInIndex()},
           {VarUpdateWithUpdaterOp::getUpdatedVarOutIndex()}}};
}

} // namespace popart
