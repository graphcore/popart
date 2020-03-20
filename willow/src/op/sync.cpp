// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <popart/op/sync.hpp>

namespace popart {

SyncOp::SyncOp(const Op::Settings &settings_, poplar::SyncType syncType)
    : Op(Onnx::CustomOperators::Sync, settings_), syncType_(syncType) {}

std::unique_ptr<Op> SyncOp::clone() const {
  return std::make_unique<SyncOp>(*this);
}

const poplar::SyncType &SyncOp::getSyncType() const { return syncType_; }

} // namespace popart
