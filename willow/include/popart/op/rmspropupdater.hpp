// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_RMSPROPUPDATER_HPP
#define GUARD_NEURALNET_RMSPROPUPDATER_HPP

#include <popart/adaptive.hpp>
#include <popart/op/varupdate.hpp>

namespace popart {

class RMSPropUpdaterOp : public Op {

public:
  RMSPropUpdaterOp(OptimizerValue eps, const Op::Settings &);

  std::unique_ptr<Op> clone() const final;
  void setup() final;

  void appendOutlineAttributes(OpSerialiserBase &) const final;

  const OptimizerValue initEps;

  static InIndex getGradInIndex() { return 0; }
  static InIndex getAccl1InIndex() { return 1; }
  static InIndex getAccl2InIndex() { return 2; }
  static InIndex getEpsInIndex() { return 3; }

  static OutIndex getUpdaterOutIndex() { return 0; }

  float getSubgraphValue() const final { return getHighSubgraphValue(); }
  virtual bool isOptimizerOp() const override { return true; }

  ReplicatedTensorShardingIndices
  getReplicatedTensorShardingIndices() const final;
};

} // namespace popart

#endif
