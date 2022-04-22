// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_SYNC_HPP
#define GUARD_NEURALNET_SYNC_HPP

#include <memory>
#include <poplar/SyncType.hpp>
#include <popart/op.hpp>

namespace popart {

class SyncOp : public Op {
public:
  SyncOp(const Op::Settings &, poplar::SyncType syncType);

  std::unique_ptr<Op> clone() const override;

  const poplar::SyncType &getSyncType() const;

  void setup() final {}

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

  bool hasSideEffect() const override { return true; }

private:
  poplar::SyncType syncType_;
};

} // namespace popart

#endif
