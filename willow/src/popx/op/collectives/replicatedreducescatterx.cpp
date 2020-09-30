// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <popart/error.hpp>
#include <popart/ir.hpp>
#include <popart/op/collectives/replicatedallgather.hpp>
#include <popart/op/collectives/replicatedreducescatter.hpp>
#include <popart/popx/devicex.hpp>
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
            outInfo(ReplicatedAllGatherOp::getOutIndex()).nelms())});
    setOutViewChangers(ReplicatedReduceScatterOp::getOutIndex(), viewChangers);

    if (!hasInViewChangers(ReplicatedReduceScatterOp::getInIndex())) {
      // Tensor not rearranged for reduceScatter yet, do it now
      auto cbr = createCollectiveBalancedReorder(toReduceScatter);
      poplar::Tensor refClone = cbr->getReferenceTensorClone(
          toReduceScatter.elementType(),
          inId(ReplicatedReduceScatterOp::getInIndex()) + "_RefClone");
      prog.add(poplar::program::Copy(toReduceScatter, refClone));
      toReduceScatter = cbr->rearrangeForCollective(refClone.flatten());
    }
  }

  poplar::OptionFlags reduceScatterOptions = dv_p->gclOptions;
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
  const auto &rrs_op = getOp<ReplicatedReduceScatterOp>();
  if (inIndex != ReplicatedReduceScatterOp::getInIndex()) {
    throw error(
        "ReplicatedReduceScatterOpx::createInput, cannot create input at {}",
        inIndex);
  }

  auto cbr = getCollectiveBalancedReorder();
  if (cbr) {
    return cbr->rearrangeForCollective(
        cbr->getReferenceTensorClone(popType(rrs_op.inTensor(inIndex)->info),
                                     name)
            .flatten());
  } else {
    throw error("ReplicatedReduceScatterOpx::createInput, "
                "CollectiveBalancedReorder not found for Op {}",
                op_p->debugName());
  }
}

std::vector<TensorId>
ReplicatedReduceScatterOpx::mustExistBeforeCreate(InIndex) const {
  auto group = getCollectiveLinkedGroup();
  TensorId mustExist =
      group.second.front()->inId(CollectivesBaseOp::getInIndex());
  logging::opx::trace(
      "ReplicatedReduceScatterOpx::mustExistBeforeCreate, must exist: {}",
      mustExist);
  return {mustExist};
}

bool ReplicatedReduceScatterOpx::hasCreatorViewChangers(InIndex index) const {
  return (index == ReplicatedReduceScatterOp::getInIndex());
}

ViewChangers
ReplicatedReduceScatterOpx::getCreatorViewChangers(InIndex index) const {
  if (index == ReplicatedReduceScatterOp::getInIndex()) {
    auto cbr = getCollectiveBalancedReorder();
    ViewChangers viewChangers(
        {std::make_shared<ReplicatedGatherOutScatterInViewChanger>(cbr)});
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
