// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <memory>
#include <set>
#include <tuple>
#include <utility>
#include <vector>
#include <popart/op/collectives/multi_replicatedreducescatter.hpp>

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

MultiReplicatedReduceScatterOp::MultiReplicatedReduceScatterOp(
    CollectiveOperator op_,
    CommGroup group_,
    const Op::Settings &settings_,
    std::vector<TensorInfo> outInfoFromBaseOps_,
    std::vector<bool> rearrangeForCollective_,
    std::vector<VGraphIdAndTileSet> inputVirtualGraphIdAndTileSet_,
    std::vector<VGraphIdAndTileSet> outputVirtualGraphIdAndTileSet_)
    : MultiCollectiveBaseOp(Onnx::CustomOperators::MultiReplicatedReduceScatter,
                            group_,
                            settings_,
                            outInfoFromBaseOps_,
                            inputVirtualGraphIdAndTileSet_,
                            outputVirtualGraphIdAndTileSet_),
      op(op_), rearrangeForCollective(rearrangeForCollective_) {}

std::unique_ptr<Op> MultiReplicatedReduceScatterOp::clone() const {
  return std::make_unique<MultiReplicatedReduceScatterOp>(*this);
}

ReplicatedTensorShardingIndices
MultiReplicatedReduceScatterOp::getReplicatedTensorShardingIndices() const {
  ReplicatedTensorShardingIndices indices;

  // All outputs are RTS tensors
  for (OutIndex outIdx = 0; outIdx < output->n(); ++outIdx) {
    indices.insert(
        std::pair<std::set<InIndex>, std::set<OutIndex>>{{}, {outIdx}});
  }
  return indices;
}

bool MultiReplicatedReduceScatterOp::rearrangeGrowPartForCollective(
    OpxGrowPartId id) const {
  return rearrangeForCollective.at(id);
}

view::Regions MultiReplicatedReduceScatterOp::modifies(InIndex index) const {
  return {view::Region::getEmpty(inRank(index))};
}

view::Regions MultiReplicatedReduceScatterOp::aliases(InIndex in,
                                                      OutIndex out) const {
  return {view::Region::getEmpty(inRank(in))};
}

} // namespace popart
