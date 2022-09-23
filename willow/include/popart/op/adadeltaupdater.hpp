// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_ADADELTAUPDATER_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_ADADELTAUPDATER_HPP_

#include <memory>

#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/optimizervalue.hpp"

namespace popart {
class OpSerialiserBase;

class AdaDeltaUpdaterOp : public Op {

public:
  AdaDeltaUpdaterOp(OptimizerValue eps, const Op::Settings &);

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
  bool isOptimizerOp() const override { return true; }

  ReplicatedTensorShardingIndices
  getReplicatedTensorShardingIndices() const final;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_ADADELTAUPDATER_HPP_
