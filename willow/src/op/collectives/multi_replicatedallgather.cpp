// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <memory>
#include <set>
#include <tuple>
#include <utility>
#include <vector>
#include <popart/op/collectives/multi_replicatedallgather.hpp>

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
#include "popart/sessionoptions.hpp"
#include "popart/tensorindex.hpp"
#include "popart/tensorinfo.hpp"
#include "popart/tensorlocation.hpp"

namespace popart {

MultiReplicatedAllGatherOp::MultiReplicatedAllGatherOp(
    CommGroup group_,
    const Op::Settings &settings_,
    std::vector<TensorInfo> outInfoFromBaseOps_,
    std::vector<bool> undoRearrangeForCollective_,
    std::vector<VGraphIdAndTileSet> inputVirtualGraphIdAndTileSet_,
    std::vector<VGraphIdAndTileSet> outputVirtualGraphIdAndTileSet_)
    : MultiCollectiveBaseOp(Onnx::CustomOperators::MultiReplicatedAllGather,
                            group_,
                            settings_,
                            outInfoFromBaseOps_,
                            inputVirtualGraphIdAndTileSet_,
                            outputVirtualGraphIdAndTileSet_),
      undoRearrangeForCollective(undoRearrangeForCollective_) {}

std::unique_ptr<Op> MultiReplicatedAllGatherOp::clone() const {
  return std::make_unique<MultiReplicatedAllGatherOp>(*this);
}

ReplicatedTensorShardingIndices
MultiReplicatedAllGatherOp::getReplicatedTensorShardingIndices() const {
  ReplicatedTensorShardingIndices indices;
  // All (n/2, same as number of outputs) inputs are RTS tensors
  for (InIndex inIdx = 0; inIdx < output->n(); ++inIdx) {
    indices.insert(
        std::pair<std::set<InIndex>, std::set<OutIndex>>{{inIdx}, {}});
  }
  return indices;
}

bool MultiReplicatedAllGatherOp::undoRearrangeGrowPartForCollective(
    OpxGrowPartId id) const {
  return undoRearrangeForCollective.at(id);
}

view::Regions MultiReplicatedAllGatherOp::modifies(InIndex index) const {
  return {view::Region::getEmpty(inRank(index))};
}

view::Regions MultiReplicatedAllGatherOp::aliases(InIndex in,
                                                  OutIndex out) const {
  return {view::Region::getEmpty(inRank(in))};
}

} // namespace popart
