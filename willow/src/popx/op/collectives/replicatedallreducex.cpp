// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <gcl/Collectives.hpp>
#include <snap/Graph.hpp>
#include <snap/Program.hpp>
#include <snap/Tensor.hpp>
#include <vector>
#include <poplar/Tensor.hpp>
#include <popart/op/collectives/replicatedallreduce.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/popx/op/collectives/replicatedallreducex.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/commgroup.hpp"
#include "popart/graphcoreoperators.hpp"
#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/op/collectives/collectives.hpp"
#include "popart/popx/op/collectives/collectivesx.hpp"
#include "popart/popx/popopx.hpp"
#include "popart/region.hpp" // IWYU pragma: keep
#include "popart/util.hpp"

namespace poplar {
class OptionFlags;
} // namespace poplar

namespace popart {

class Op;

namespace popx {

ReplicatedAllReduceOpx::ReplicatedAllReduceOpx(Op *op, Devicex *devicex)
    : CollectivesBaseOpx(op, devicex) {
  verifyOp<CollectivesBaseOp>(op);
}

void ReplicatedAllReduceOpx::grow(snap::program::Sequence &prog) const {
  const auto &rarOp = getOp<ReplicatedAllReduceOp>();

  const auto inIndex      = ReplicatedAllReduceOp::getInIndex();
  poplar::Tensor toReduce = getInTensor(inIndex).getPoplarTensor();
  const poplar::OptionFlags &allReduceOptions = dv_p->lowering().gclOptions;
  poplar::Tensor output                       = gcl::allReduceCrossReplica(
      graph().getPoplarGraph(),
      toReduce,
      getPoplarCollectiveOperator(rarOp.getCollectiveOp()),
      prog.getPoplarSequence(),
      toGCLCommGroup(rarOp.getGCLCommGroup()),
      debugContext("replicatedAllReduce"),
      allReduceOptions);

  logging::transform::trace("[ReplicatedAllReduceOpx::grow] comm group: {}, "
                            "input shape: {}, output shape: {}",
                            rarOp.getGCLCommGroup(),
                            toReduce.shape(),
                            output.shape());

  if (hasInViewChangers(ReplicatedAllReduceOp::getInIndex())) {
    setOutViewChangers(ReplicatedAllReduceOp::getOutIndex(),
                       getInViewChangers(ReplicatedAllReduceOp::getInIndex()));
  }
  setOutTensor(ReplicatedAllReduceOp::getOutIndex(),
               snap::Tensor{output, graph()});
}

InputCreatorType ReplicatedAllReduceOpx::getInputCreatorType(InIndex) const {
  return InputCreatorType::CanUnwind;
}

snap::Tensor ReplicatedAllReduceOpx::unwindTensorLayout(snap::Tensor tensor,
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

void ReplicatedAllReduceInplaceOpx::grow(snap::program::Sequence &prog) const {
  const auto &rarOp = getOp<ReplicatedAllReduceOp>();

  const auto inIndex      = ReplicatedAllReduceInplaceOp::getInIndex();
  poplar::Tensor toReduce = getInTensor(inIndex).getPoplarTensor();
  const poplar::OptionFlags &allReduceOptions = dv_p->lowering().gclOptions;
  auto inputShape                             = toReduce.shape();

  gcl::allReduceInPlaceCrossReplica(
      graph().getPoplarGraph(),
      toReduce,
      getPoplarCollectiveOperator(rarOp.getCollectiveOp()),
      prog.getPoplarSequence(),
      toGCLCommGroup(rarOp.getGCLCommGroup()),
      debugContext("replicatedAllReduce"),
      allReduceOptions);

  auto outputShape = toReduce.shape();

  logging::transform::trace("[ReplicatedAllReduceOpx::grow] comm group: {}, "
                            "input shape: {}, output shape: {}",
                            rarOp.getGCLCommGroup(),
                            inputShape,
                            outputShape);

  if (hasInViewChangers(ReplicatedAllReduceOp::getInIndex())) {
    setOutViewChangers(ReplicatedAllReduceOp::getOutIndex(),
                       getInViewChangers(ReplicatedAllReduceOp::getInIndex()));
  }
  setOutTensor(ReplicatedAllReduceInplaceOp::getOutIndex(),
               snap::Tensor{toReduce, graph()});
}

namespace {
OpxCreator<ReplicatedAllReduceInplaceOpx> ReplicatedAllReduceInplaceOpxCreator(
    Onnx::CustomOperators::ReplicatedAllReduceInplace);

OpxCreator<ReplicatedAllReduceOpx>
    ReplicatedAllReduceOpxCreator(Onnx::CustomOperators::ReplicatedAllReduce);

} // namespace

} // namespace popx
} // namespace popart
