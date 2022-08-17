// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_COLLECTIVES_REPLICATEDREDUCESCATTER_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_COLLECTIVES_REPLICATEDREDUCESCATTER_HPP_

#include <memory>
#include <tuple>
#include <popart/op/collectives/collectives.hpp>

#include "popart/names.hpp"
#include "popart/op.hpp"

namespace popart {
class AliasModel;
class CommGroup;
class ReplicaGrouping;
class OpSerialiserBase;
class ReplicaEqualAnalysisProxy;
struct OperatorIdentifier;

class ReplicatedReduceScatterOp : public CollectivesBaseOp {
public:
  // TODO(T67766): Delete.
  [[deprecated]] ReplicatedReduceScatterOp(
      const OperatorIdentifier &,
      CollectiveOperator op,
      CommGroup group,
      bool configureOutputForReplicatedTensorSharding,
      const Op::Settings &);
  // TODO(T67766): Delete.
  [[deprecated]] ReplicatedReduceScatterOp(const OperatorIdentifier &,
                                           CollectiveOperator op,
                                           CommGroup group,
                                           const Op::Settings &);
  ReplicatedReduceScatterOp(const OperatorIdentifier &,
                            CollectiveOperator op,
                            const ReplicaGrouping &grouping,
                            bool configureOutputForReplicatedTensorSharding,
                            const Op::Settings &);
  ReplicatedReduceScatterOp(const OperatorIdentifier &,
                            CollectiveOperator op,
                            const ReplicaGrouping &grouping,
                            const Op::Settings &);
  ReplicatedReduceScatterOp(const OperatorIdentifier &, const Op::Settings &);

  std::unique_ptr<Op> clone() const final;
  void setup() final;
  float getSubgraphValue() const final { return getHighSubgraphValue(); }
  void appendOutlineAttributes(OpSerialiserBase &) const override;
  CollectiveOperator getCollectiveOp() const { return op; }

  ReplicatedTensorShardingIndices
  getReplicatedTensorShardingIndices() const override;

  /**
   * Check \a RTS mode (see collectives.hpp)
   * \return True if this operation is configured for replicated tensor sharding
   */
  bool isConfigureOutputForReplicatedTensorSharding() const override;

  std::tuple<ReplEqOutputMap, ReplEqModifiedInputMap>
  fwdPropagateIsReplicaEqual(const AliasModel &aliasModel,
                             const ReplEqInputMap &inputMap,
                             ReplicaEqualAnalysisProxy &proxy) const override;

protected:
  CollectiveOperator op;
  /**
   * If enabled, configures the Op for replicated tensor sharding
   */
  bool configureOutputForReplicatedTensorSharding;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_COLLECTIVES_REPLICATEDREDUCESCATTER_HPP_
