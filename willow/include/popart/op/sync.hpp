// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_SYNC_HPP
#define GUARD_NEURALNET_SYNC_HPP

#include <popart/op.hpp>

#include <poplar/SyncType.hpp>

namespace popart {

class SyncOp : public Op {
public:
  SyncOp(const Op::Settings &, poplar::SyncType syncType);

  std::unique_ptr<Op> clone() const override;

  const poplar::SyncType &getSyncType() const;

  void setup() final {}

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

private:
  poplar::SyncType syncType_;
};

} // namespace popart

#endif
