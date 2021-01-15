// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_REPLICATEDALLREDUCE_HPP
#define GUARD_NEURALNET_REPLICATEDALLREDUCE_HPP

#include <popart/op/collectives/collectives.hpp>

namespace popart {

class ReplicatedAllReduceOp : public CollectivesBaseOp {
public:
  ReplicatedAllReduceOp(const OperatorIdentifier &_opid,
                        CollectiveOperator op_,
                        const Op::Settings &settings_);
  ReplicatedAllReduceOp(const OperatorIdentifier &_opid,
                        const Op::Settings &settings_);

  std::unique_ptr<Op> clone() const override;
  virtual std::unique_ptr<Op>
  getInplaceVariant(const OperatorIdentifier &) const override;
  void setup() override;
  float getSubgraphValue() const final { return getHighSubgraphValue(); }
  void appendOutlineAttributes(OpSerialiserBase &) const override;
  CollectiveOperator getCollectiveOp() const { return op; }

protected:
  CollectiveOperator op;
};

class ReplicatedAllReduceInplaceOp : public ReplicatedAllReduceOp {
public:
  ReplicatedAllReduceInplaceOp(const OperatorIdentifier &_opid,
                               CollectiveOperator op_,
                               const Op::Settings &settings_);
  ReplicatedAllReduceInplaceOp(const OperatorIdentifier &_opid,
                               const Op::Settings &settings_);
  ReplicatedAllReduceInplaceOp(const ReplicatedAllReduceOp &);

  view::Regions modifies(InIndex) const override;
  view::Regions aliases(InIndex, OutIndex) const override;
  std::unique_ptr<Op> clone() const final;
  void setup() final;
};

} // namespace popart

#endif
