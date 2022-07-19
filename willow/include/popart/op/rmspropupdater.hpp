// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_RMSPROPUPDATER_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_RMSPROPUPDATER_HPP_

#include <memory>

#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/optimizervalue.hpp"

namespace popart {
class OpSerialiserBase;

class RMSPropUpdaterOp : public Op {

public:
  RMSPropUpdaterOp(OptimizerValue eps, bool TFVariant, const Op::Settings &);

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

  const bool TFVariant;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_RMSPROPUPDATER_HPP_
