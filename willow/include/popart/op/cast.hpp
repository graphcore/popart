// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_CAST_HPP
#define GUARD_NEURALNET_CAST_HPP

#include <popart/op.hpp>

namespace popart {

class CastOp : public Op {
public:
  CastOp(const OperatorIdentifier &_opid,
         DataType _to,
         const Op::Settings &settings);
  std::unique_ptr<Op> clone() const override;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  void setup() override;

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }

  DataType toDataType() { return to; }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

  bool canShard() const override { return true; }

  ReplicatedTensorShardingIndices
  getReplicatedTensorShardingIndices() const override {
    return {{{CastOp::getInIndex()}, {CastOp::getOutIndex()}}};
  }

private:
  DataType to;
};

class CastGradOp : public CastOp {
public:
  CastGradOp(const CastOp &fwdOp);
  std::unique_ptr<Op> clone() const final;

  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &
  gradOutToNonGradIn() const final; // No input from fwd pass??
};

} // namespace popart

#endif
