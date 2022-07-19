// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_COLLECTIVES_REPLICATEDALLREDUCE_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_COLLECTIVES_REPLICATEDALLREDUCE_HPP_

#include <memory>
#include <tuple>
#include <popart/op/collectives/collectives.hpp>

#include "popart/error.hpp"
#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"

namespace popart {
class AliasModel;
class CommGroup;
class OpSerialiserBase;
class ReplicaEqualAnalysisProxy;
class Tensor;
struct OperatorIdentifier;

class ReplicatedAllReduceOp : public CollectivesBaseOp {
public:
  ReplicatedAllReduceOp(const OperatorIdentifier &,
                        CollectiveOperator op,
                        CommGroup group,
                        const Op::Settings &);
  ReplicatedAllReduceOp(const OperatorIdentifier &, const Op::Settings &);

  std::unique_ptr<Op> clone() const override;
  virtual std::unique_ptr<Op>
  getInplaceVariant(const OperatorIdentifier &) const override;
  void setup() override;
  float getSubgraphValue() const final { return getHighSubgraphValue(); }
  void appendOutlineAttributes(OpSerialiserBase &) const override;
  CollectiveOperator getCollectiveOp() const { return op; }
  virtual void growAliasModel(AliasModel &) const override;

  ReplicatedTensorShardingIndices
  getReplicatedTensorShardingIndices() const override;

  bool hasCorrespondingLinkedIndexTensor(Tensor *t) override { return false; }
  Tensor *getCorrespondingLinkedIndexTensor(Tensor *t) override {
    throw error("AllReduce does not support linked index tensors");
  }
  bool isCollectiveLinkedIndexTensor(InIndex in) const override {
    return false;
  }
  bool isCollectiveLinkedIndexTensor(Tensor *t) const override { return false; }

  std::tuple<ReplEqOutputMap, ReplEqModifiedInputMap>
  fwdPropagateIsReplicaEqual(const AliasModel &aliasModel,
                             const ReplEqInputMap &inputMap,
                             ReplicaEqualAnalysisProxy &proxy) const override;

protected:
  CollectiveOperator op;
};

class ReplicatedAllReduceInplaceOp : public ReplicatedAllReduceOp {
public:
  ReplicatedAllReduceInplaceOp(const OperatorIdentifier &_opid,
                               CollectiveOperator op_,
                               CommGroup group,
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

#endif // POPART_WILLOW_INCLUDE_POPART_OP_COLLECTIVES_REPLICATEDALLREDUCE_HPP_
