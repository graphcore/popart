// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <popart/error.hpp>
#include <popart/ir.hpp>
#include <popart/op/collectives/replicatedallgather.hpp>
#include <popart/op/collectives/replicatedreducescatter.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/popx/op/collectives/collectivesx.hpp>
#include <popart/popx/op/collectives/replicatedallgatherx.hpp>
#include <popart/popx/op/collectives/replicatedreducescatterx.hpp>
#include <popart/popx/opxmanager.hpp>

#include <gcl/Collectives.hpp>

namespace popart {
namespace popx {

ReplicatedReduceScatterOpx::ReplicatedReduceScatterOpx(Op *op, Devicex *devicex)
    : CollectivesBaseOpx(op, devicex) {
  verifyOp<ReplicatedReduceScatterOp>(
      op, Onnx::CustomOperators::ReplicatedReduceScatter);
}

void ReplicatedReduceScatterOpx::grow(poplar::program::Sequence &prog) const {
  const auto inIndex             = ReplicatedReduceScatterOp::getInIndex();
  poplar::Tensor toReduceScatter = getInTensor(inIndex);

  if (hasInput(ReplicatedAllGatherOp::getCollectiveLinkedIndex())) {
    ViewChangers viewChangers(
        {std::make_shared<ReplicatedGatherInScatterOutViewChanger>(
            outInfo(ReplicatedAllGatherOp::getOutIndex()).nelms(),
            getCollectiveLinkedGroup().first)});
    setOutViewChangers(ReplicatedReduceScatterOp::getOutIndex(), viewChangers);

    if (!hasInViewChangers(ReplicatedReduceScatterOp::getInIndex())) {
      // Tensor not rearranged for reduceScatter yet, do it now
      auto cbr = createCollectiveBalancedReorder(toReduceScatter);
      auto c   = cbr->createCollectivesTensor(
          toReduceScatter.elementType(),
          inId(ReplicatedReduceScatterOp::getInIndex()));
      auto ref = cbr->undoRearrangeForCollective(c);
      prog.add(poplar::program::Copy(toReduceScatter.flatten(), ref.flatten()));
      toReduceScatter = c;
    }
  }

  poplar::OptionFlags reduceScatterOptions = dv_p->lowering().gclOptions;
  reduceScatterOptions.set("useReplicatedImplementation", "true");

  poplar::Tensor reducedScattered =
      gcl::reduceScatter(graph(),
                         toReduceScatter.flatten(),
                         popops::Operation::ADD,
                         prog,
                         "",
                         reduceScatterOptions);

  setOutTensor(ReplicatedReduceScatterOp::getOutIndex(), reducedScattered);
}

InputCreatorType
ReplicatedReduceScatterOpx::getInputCreatorType(InIndex index) const {
  return index == ReplicatedReduceScatterOp::getInIndex() &&
                 hasInput(ReplicatedAllGatherOp::getCollectiveLinkedIndex())
             ? InputCreatorType::CanCreate
             : Opx::getInputCreatorType(index);
}

poplar::Tensor
ReplicatedReduceScatterOpx::createInput(int inIndex,
                                        const std::string &name) const {
  if (inIndex != ReplicatedReduceScatterOp::getInIndex()) {
    throw error(
        "ReplicatedReduceScatterOpx::createInput, cannot create input at {}",
        inIndex);
  }

  auto cbr = getCollectiveBalancedReorder();
  if (!cbr) {
    throw error("ReplicatedReduceScatterOpx::createInput, "
                "CollectiveBalancedReorder not found for Op {}",
                op_p->debugName());
  }

  const auto &rrs_op = getOp<ReplicatedReduceScatterOp>();
  const auto &type   = popType(rrs_op.inTensor(inIndex)->info);
  return cbr->createCollectivesTensor(type, name);
}

std::vector<TensorId>
ReplicatedReduceScatterOpx::mustExistBeforeCreate(InIndex) const {
  const auto &rrs_op              = getOp<ReplicatedReduceScatterOp>();
  auto group                      = getCollectiveLinkedGroup();
  std::vector<TensorId> mustExist = {
      group.second.front()->inId(CollectivesBaseOp::getInIndex()),
      group.second.front()->outId(CollectivesBaseOp::getOutIndex())};
  logging::opx::trace("ReplicatedReduceScatterOpx::mustExistBeforeCreate, Op "
                      "{}, must exist: {}",
                      rrs_op.debugName(),
                      mustExist);
  return mustExist;
}

bool ReplicatedReduceScatterOpx::hasCreatorViewChangers(InIndex index) const {
  return (index == ReplicatedReduceScatterOp::getInIndex());
}

ViewChangers
ReplicatedReduceScatterOpx::getCreatorViewChangers(InIndex index) const {
  if (index == ReplicatedReduceScatterOp::getInIndex()) {
    auto cbr = getCollectiveBalancedReorder();
    ViewChangers viewChangers(
        {std::make_shared<ReplicatedGatherOutScatterInViewChanger>(
            cbr, getCollectiveLinkedGroup().first)});
    return viewChangers;
  }
  throw error(
      "ReplicatedReduceScatterOpx::getCreatorViewChangers: Invalid index = " +
      std::to_string(index));
}

namespace {

OpxCreator<ReplicatedReduceScatterOpx> ReplicatedReduceScatterOpxCreator(
    Onnx::CustomOperators::ReplicatedReduceScatter);

} // namespace

} // namespace popx
} // namespace popart
