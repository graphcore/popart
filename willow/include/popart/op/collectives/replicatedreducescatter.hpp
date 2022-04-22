// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_REPLICATEDREDUCESCATTER_HPP
#define GUARD_NEURALNET_REPLICATEDREDUCESCATTER_HPP

#include <memory>
#include <tuple>
#include <popart/op/collectives/collectives.hpp>

#include "popart/names.hpp"
#include "popart/op.hpp"

namespace popart {
class AliasModel;
class CommGroup;
class OpSerialiserBase;
class ReplicaEqualAnalysisProxy;
struct OperatorIdentifier;

class ReplicatedReduceScatterOp : public CollectivesBaseOp {
public:
  ReplicatedReduceScatterOp(const OperatorIdentifier &,
                            CollectiveOperator op,
                            CommGroup group,
                            bool configureOutputForReplicatedTensorSharding,
                            const Op::Settings &);
  ReplicatedReduceScatterOp(const OperatorIdentifier &,
                            CollectiveOperator op,
                            CommGroup group,
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

#endif
