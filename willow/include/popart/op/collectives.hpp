// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_COLLECTIVES_HPP
#define GUARD_NEURALNET_COLLECTIVES_HPP

#include <popart/op.hpp>

namespace popart {

class CollectivesBaseOp : public Op {
public:
  CollectivesBaseOp(const OperatorIdentifier &opid,
                    const Op::Settings &settings);

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }
};

class ReplicatedAllReduceOp : public CollectivesBaseOp {
public:
  ReplicatedAllReduceOp(const OperatorIdentifier &opid,
                        const Op::Settings &settings);

  std::unique_ptr<Op> clone() const final;
  virtual std::unique_ptr<Op>
  getInplaceVariant(const OperatorIdentifier &) const;
  void setup() final;
  float getSubgraphValue() const final { return getHighSubgraphValue(); }
};

class ReplicatedAllReduceInplaceOp : public CollectivesBaseOp {
public:
  ReplicatedAllReduceInplaceOp(const OperatorIdentifier &opid,
                               const Op::Settings &settings);
  ReplicatedAllReduceInplaceOp(const ReplicatedAllReduceOp &);

  view::Regions modifies(InIndex) const override;
  view::Regions aliases(InIndex, OutIndex) const override;
  std::unique_ptr<Op> clone() const final;
  void setup() final;
  float getSubgraphValue() const final { return getHighSubgraphValue(); }
};

class ReplicatedReduceScatterOp : public CollectivesBaseOp {
public:
  ReplicatedReduceScatterOp(const OperatorIdentifier &opid,
                            const Op::Settings &settings);
  std::unique_ptr<Op> clone() const final;
  void setup() final;
  float getSubgraphValue() const final { return getHighSubgraphValue(); }
};

} // namespace popart

#endif
