// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <gcl/Collectives.hpp>
#include <vector>
#include <poplar/Tensor.hpp>
#include <popart/op/collectives/replicatedallreduce.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/popx/op/collectives/replicatedallreducex.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/graphcoreoperators.hpp"
#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/op/collectives/collectives.hpp"
#include "popart/popx/op/collectives/collectivesx.hpp"
#include "popart/popx/opx.hpp"
#include "popart/region.hpp" // IWYU pragma: keep
#include "popart/replicagrouping.hpp"
#include "popart/util.hpp"

namespace poplar {
class OptionFlags;
namespace program {
class Sequence;
} // namespace program
} // namespace poplar

namespace popart {

class Op;

namespace popx {

ReplicatedAllReduceOpx::ReplicatedAllReduceOpx(Op *op, Devicex *devicex)
    : CollectivesBaseOpx(op, devicex) {
  verifyOp<CollectivesBaseOp>(op);
}

void ReplicatedAllReduceOpx::grow(poplar::program::Sequence &prog) const {
  const auto &rarOp = getOp<ReplicatedAllReduceOp>();

  const auto inIndex      = ReplicatedAllReduceOp::getInIndex();
  poplar::Tensor toReduce = getInTensor(inIndex);
  const poplar::OptionFlags &allReduceOptions = dv_p->lowering().gclOptions;
  poplar::Tensor output                       = gcl::allReduceCrossReplica(
      graph(),
      toReduce,
      getPoplarCollectiveOperator(rarOp.getCollectiveOp()),
      prog,
      toGclCommGroup(rarOp.getReplicaGrouping()),
      debugContext("replicatedAllReduce"),
      allReduceOptions);

  logging::transform::trace("[ReplicatedAllReduceOpx::grow] replica grouping: "
                            "{}, input shape: {}, output shape: {}",
                            rarOp.getReplicaGrouping(),
                            toReduce.shape(),
                            output.shape());

  if (hasInViewChangers(ReplicatedAllReduceOp::getInIndex())) {
    setOutViewChangers(ReplicatedAllReduceOp::getOutIndex(),
                       getInViewChangers(ReplicatedAllReduceOp::getInIndex()));
  }
  setOutTensor(ReplicatedAllReduceOp::getOutIndex(), output);
}

InputCreatorType ReplicatedAllReduceOpx::getInputCreatorType(InIndex) const {
  return InputCreatorType::CanUnwind;
}

poplar::Tensor ReplicatedAllReduceOpx::unwindTensorLayout(poplar::Tensor tensor,
                                                          InIndex,
                                                          OutIndex) const {
  return tensor;
}

view::RegMap ReplicatedAllReduceOpx::unwindRegion(InIndex, OutIndex) const {
  return [](const view::Region &r) { return view::Regions(1, r); };
}

ReplicatedAllReduceInplaceOpx::ReplicatedAllReduceInplaceOpx(Op *op,
                                                             Devicex *devicex)
    : ReplicatedAllReduceOpx(op, devicex) {
  verifyOp<ReplicatedAllReduceInplaceOp>(
      op, Onnx::CustomOperators::ReplicatedAllReduceInplace);
}

void ReplicatedAllReduceInplaceOpx::grow(
    poplar::program::Sequence &prog) const {
  const auto &rarOp = getOp<ReplicatedAllReduceOp>();

  const auto inIndex      = ReplicatedAllReduceInplaceOp::getInIndex();
  poplar::Tensor toReduce = getInTensor(inIndex);
  const poplar::OptionFlags &allReduceOptions = dv_p->lowering().gclOptions;
  auto inputShape                             = toReduce.shape();

  gcl::allReduceInPlaceCrossReplica(
      graph(),
      toReduce,
      getPoplarCollectiveOperator(rarOp.getCollectiveOp()),
      prog,
      toGclCommGroup(rarOp.getReplicaGrouping()),
      debugContext("replicatedAllReduce"),
      allReduceOptions);

  auto outputShape = toReduce.shape();

  logging::transform::trace("[ReplicatedAllReduceOpx::grow] replica grouping: "
                            "{}, input shape: {}, output shape: {}",
                            rarOp.getReplicaGrouping(),
                            inputShape,
                            outputShape);

  if (hasInViewChangers(ReplicatedAllReduceOp::getInIndex())) {
    setOutViewChangers(ReplicatedAllReduceOp::getOutIndex(),
                       getInViewChangers(ReplicatedAllReduceOp::getInIndex()));
  }
  setOutTensor(ReplicatedAllReduceInplaceOp::getOutIndex(), toReduce);
}

namespace {
OpxCreator<ReplicatedAllReduceInplaceOpx> ReplicatedAllReduceInplaceOpxCreator(
    Onnx::CustomOperators::ReplicatedAllReduceInplace);

OpxCreator<ReplicatedAllReduceOpx>
    ReplicatedAllReduceOpxCreator(Onnx::CustomOperators::ReplicatedAllReduce);

} // namespace

} // namespace popx
} // namespace popart
