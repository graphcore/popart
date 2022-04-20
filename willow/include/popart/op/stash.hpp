// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_STASH_HPP
#define GUARD_NEURALNET_STASH_HPP

#include <cstdint>
#include <memory>
#include <popart/op.hpp>

#include "popart/names.hpp"

namespace popart {
class OpSerialiserBase;
struct OperatorIdentifier;

class StashOp : public Op {
public:
  StashOp(const OperatorIdentifier &, int64_t stashSize_, const Op::Settings &);

  std::unique_ptr<Op> clone() const override;
  void setup() final;

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }

  int64_t getStashSize();
  TensorId getStashedTensorId() const;

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

  void appendOutlineAttributes(OpSerialiserBase &) const override;

  bool isOutlineable() const override { return false; }

private:
  int64_t stashSize;
};

} // namespace popart

#endif
