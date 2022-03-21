// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <popart/op/collectives/multi_replicatedallreduce.hpp>

namespace popart {

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
                            modifiesIndexInplace_,
                            outInfoFromBaseOps_,
                            inputVirtualGraphIdAndTileSet_,
                            outputVirtualGraphIdAndTileSet_),
      op(op_) {}

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
  if (getModifiesIndexInplace().at(index)) {
    return {view::Region::getFull(inShape(index))};
  }
  return {view::Region::getEmpty(inRank(index))};
}

view::Regions MultiReplicatedAllReduceOp::aliases(InIndex in,
                                                  OutIndex out) const {
  if (in == out && getModifiesIndexInplace().at(in)) {
    return {view::Region::getFull(inShape(in))};
  } else {
    return {view::Region::getEmpty(inRank(in))};
  }
}

void MultiReplicatedAllReduceOp::growAliasModel(AliasModel &m) const {
  growAliasModelMulti(m);
}

} // namespace popart
