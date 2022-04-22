// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ADAMUPDATER_HPP
#define GUARD_NEURALNET_ADAMUPDATER_HPP

#include <memory>
#include <popart/adam.hpp>

#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/optimizervalue.hpp"

namespace popart {
class OpSerialiserBase;

class AdamUpdaterOp : public Op {

public:
  AdamUpdaterOp(AdamMode mode_,
                OptimizerValue wd,
                OptimizerValue b1,
                OptimizerValue b2,
                OptimizerValue eps,
                const Op::Settings &);

  std::unique_ptr<Op> clone() const final;
  void setup() final;

  void appendOutlineAttributes(OpSerialiserBase &) const final;

  AdamMode mode;

  const OptimizerValue initWd;
  const OptimizerValue initB1;
  const OptimizerValue initB2;
  const OptimizerValue initEps;

  static InIndex getVarInIndex() { return 0; }
  static InIndex getAccl1InIndex() { return 1; }
  static InIndex getAccl2InIndex() { return 2; }
  static InIndex getStepInIndex() { return 3; }
  static InIndex getWdInIndex() { return 4; }
  static InIndex getBeta1InIndex() { return 5; }
  static InIndex getBeta2InIndex() { return 6; }
  static InIndex getEpsInIndex() { return 7; }

  static OutIndex getUpdaterOutIndex() { return 0; }

  // Opx implementation has heavy usage of popops::expr that can result in large
  // code sizes. Outlining these Ops can reduce the impact of using Adam-based
  // optimizers.
  float getSubgraphValue() const final { return getHighSubgraphValue(); }
  virtual bool isOptimizerOp() const override { return true; }

  view::Regions modifies(InIndex) const final;

  ReplicatedTensorShardingIndices
  getReplicatedTensorShardingIndices() const final;
};

} // namespace popart

#endif
