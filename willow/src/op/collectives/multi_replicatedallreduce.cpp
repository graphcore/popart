// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <memory>
#include <set>
#include <tuple>
#include <utility>
#include <vector>
#include <popart/op/collectives/multi_replicatedallreduce.hpp>

#include "popart/analysis/replicaequal/replicaequalanalysisproxy.hpp"
#include "popart/commgroup.hpp"
#include "popart/error.hpp"
#include "popart/graphcoreoperators.hpp"
#include "popart/ir.hpp"
#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/op/collectives/collectives.hpp"
#include "popart/region.hpp"
#include "popart/replicagrouping.hpp"
#include "popart/sessionoptions.hpp"
#include "popart/tensorindex.hpp"
#include "popart/tensorinfo.hpp"
#include "popart/tensorlocation.hpp"

namespace popart {
class AliasModel;
class Tensor;

MultiReplicatedAllReduceOp::MultiReplicatedAllReduceOp(
    CollectiveOperator op_,
    CommGroup group_,
    const Op::Settings &settings_,
    std::vector<bool> modifiesIndexInplace_,
    std::vector<TensorInfo> outInfoFromBaseOps_,
    std::vector<VGraphIdAndTileSet> inputVirtualGraphIdAndTileSet_,
    std::vector<VGraphIdAndTileSet> outputVirtualGraphIdAndTileSet_)
    : MultiCollectiveBaseOp(Onnx::CustomOperators::MultiReplicatedAllReduce,
                            group_,
                            settings_,
                            outInfoFromBaseOps_,
                            inputVirtualGraphIdAndTileSet_,
                            outputVirtualGraphIdAndTileSet_),
      op(op_), modifiesIndexInplace(modifiesIndexInplace_) {}

MultiReplicatedAllReduceOp::MultiReplicatedAllReduceOp(
    CollectiveOperator op_,
    const ReplicaGrouping &grouping,
    const Op::Settings &settings_,
    const std::vector<bool> &modifiesIndexInplace_,
    const std::vector<TensorInfo> &outInfoFromBaseOps_,
    const std::vector<VGraphIdAndTileSet> &inputVirtualGraphIdAndTileSet_,
    const std::vector<VGraphIdAndTileSet> &outputVirtualGraphIdAndTileSet_)
    : MultiCollectiveBaseOp(Onnx::CustomOperators::MultiReplicatedAllReduce,
                            grouping,
                            settings_,
                            outInfoFromBaseOps_,
                            inputVirtualGraphIdAndTileSet_,
                            outputVirtualGraphIdAndTileSet_),
      op(op_), modifiesIndexInplace(modifiesIndexInplace_) {}

std::unique_ptr<Op> MultiReplicatedAllReduceOp::clone() const {
  return std::make_unique<MultiReplicatedAllReduceOp>(*this);
}

bool MultiReplicatedAllReduceOp::hasCorrespondingLinkedIndexTensor(Tensor *t) {
  return false;
}

Tensor *
MultiReplicatedAllReduceOp::getCorrespondingLinkedIndexTensor(Tensor *t) {
  throw error("[MultiReplicatedAllReduceOp::getCorrespondingLinkedIndexTensor] "
              "MultiAllReduce does not support linked index tensors");
}

bool MultiReplicatedAllReduceOp::isCollectiveLinkedIndexTensor(
    InIndex in) const {
  return false;
}

bool MultiReplicatedAllReduceOp::isCollectiveLinkedIndexTensor(
    Tensor *t) const {
  return false;
}

ReplicatedTensorShardingIndices
MultiReplicatedAllReduceOp::getReplicatedTensorShardingIndices() const {
  ReplicatedTensorShardingIndices indices;
  for (OutIndex outIdx = 0; outIdx < output->n(); ++outIdx) {
    indices.insert(std::pair<std::set<InIndex>, std::set<OutIndex>>{
        {(InIndex)outIdx}, {outIdx}});
  }
  return indices;
}

view::Regions MultiReplicatedAllReduceOp::modifies(InIndex index) const {
  if (modifiesIndexInplace.at(index)) {
    return {view::Region::getFull(inShape(index))};
  }
  return {view::Region::getEmpty(inRank(index))};
}

view::Regions MultiReplicatedAllReduceOp::aliases(InIndex in,
                                                  OutIndex out) const {
  if (in == out && modifiesIndexInplace.at(in)) {
    return {view::Region::getFull(inShape(in))};
  } else {
    return {view::Region::getEmpty(inRank(in))};
  }
}

void MultiReplicatedAllReduceOp::growAliasModel(AliasModel &m) const {
  growAliasModelMulti(m);
}

std::tuple<ReplEqOutputMap, ReplEqModifiedInputMap>
MultiReplicatedAllReduceOp::fwdPropagateIsReplicaEqual(
    const AliasModel &aliasModel,
    const ReplEqInputMap &inputMap,
    ReplicaEqualAnalysisProxy &proxy) const {

  // TODO(T51589): Amend logic to be more fine-grained, taking into account
  // CommGroup settings. We should work out replica-equalness over subsets
  // of replicas instead instead of having only tracking if a tensor is
  // replica-equal for all replicas or not.

  const auto numReplicas                = getReplicaGrouping().getNumReplicas();
  const auto groupSize                  = getReplicaGrouping().getGroupSize();
  const auto isLocal                    = (op == CollectiveOperator::Local);
  const auto isReductionOverAllReplicas = numReplicas == groupSize;

  // For all reduction methods except Local, the output should be identical
  // across replicas within a group. So outputs are equal across all replicas
  // only if the grouping includes all replicas.
  if (!isLocal && isReductionOverAllReplicas) {
    ReplEqOutputMap result;
    for (OutIndex out = 0; out < output->n(); out++) {
      result[out] = true;
    }
    return {result, proxy.getModifiedInputMapFromAliases(this, result)};
  } else {
    return Op::fwdPropagateIsReplicaEqual(aliasModel, inputMap, proxy);
  }
}

} // namespace popart
