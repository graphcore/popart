// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_COLLECTIVES_REPLICATEDALLGATHER_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_COLLECTIVES_REPLICATEDALLGATHER_HPP_

#include <memory>
#include <tuple>
#include <popart/op/collectives/collectives.hpp>

#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/tensorinfo.hpp"

namespace popart {
class AliasModel;
class CommGroup;
class ReplicaGrouping;
class ReplicaEqualAnalysisProxy;
struct OperatorIdentifier;

class ReplicatedAllGatherOp : public CollectivesBaseOp {
public:
  // TODO(T67766): Delete.
  [[deprecated]] ReplicatedAllGatherOp(const OperatorIdentifier &,
                                       CommGroup group,
                                       const Op::Settings &);
  // TODO(T67766): Delete.
  [[deprecated]] ReplicatedAllGatherOp(const OperatorIdentifier &,
                                       CommGroup group,
                                       const Op::Settings &,
                                       TensorInfo outInfo);
  ReplicatedAllGatherOp(const OperatorIdentifier &,
                        const ReplicaGrouping &grouping,
                        const Op::Settings &);
  ReplicatedAllGatherOp(const OperatorIdentifier &,
                        const ReplicaGrouping &grouping,
                        const Op::Settings &,
                        const TensorInfo &outInfo);

  std::unique_ptr<Op> clone() const final;
  void setup() final;

  float getSubgraphValue() const final { return getHighSubgraphValue(); }

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

private:
  TensorInfo gatheredOutInfo;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_COLLECTIVES_REPLICATEDALLGATHER_HPP_
