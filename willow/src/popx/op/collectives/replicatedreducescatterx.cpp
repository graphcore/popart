// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <popops/Zero.hpp>
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

void ReplicatedReduceScatterOpx::grow(snap::program::Sequence &prog) const {
  const auto &rrsOp = getOp<ReplicatedReduceScatterOp>();

  const auto inIndex   = ReplicatedReduceScatterOp::getInIndex();
  auto toReduceScatter = getInTensor(inIndex);

  if (hasInput(ReplicatedAllGatherOp::getCollectiveLinkedIndex())) {
    ViewChangers viewChangers(
        {std::make_shared<ReplicatedGatherInScatterOutViewChanger>(
            outInfo(ReplicatedAllGatherOp::getOutIndex()).nelms(),
            getCollectiveLinkedGroup().first)});
    setOutViewChangers(ReplicatedReduceScatterOp::getOutIndex(), viewChangers);

    if (!hasInViewChangers(ReplicatedReduceScatterOp::getInIndex()) ||
        getInViewChangers(ReplicatedReduceScatterOp::getInIndex()) !=
            viewChangers) {
      logging::opx::trace("ReplicatedReduceScatterOpx::grow rearranging {}",
                          inId(ReplicatedReduceScatterOp::getInIndex()));

      // Tensor not rearranged for reduceScatter yet, do it now
      auto cbr = createCollectiveBalancedReorder(toReduceScatter);
      auto c   = snap::Tensor{cbr->createCollectivesTensor(
                                toReduceScatter.elementType(),
                                inId(ReplicatedReduceScatterOp::getInIndex())),
                            graph()};
      popops::zero(graph().getPoplarGraph(),
                   c.getPoplarTensor(),
                   prog.getPoplarSequence(),
                   debugContext());
      auto ref = cbr->undoRearrangeForCollective(c.getPoplarTensor());
      if (hasInViewChangers(ReplicatedReduceScatterOp::getInIndex())) {
        prog.getPoplarSequence().add(poplar::program::Copy(
            getInViewChangers(ReplicatedReduceScatterOp::getInIndex())
                .apply(toReduceScatter)
                .getPoplarTensor()
                .flatten(),
            ref.flatten(),
            false,
            debugContext()));
      } else {
        prog.getPoplarSequence().add(
            poplar::program::Copy(toReduceScatter.flatten().getPoplarTensor(),
                                  ref.flatten(),
                                  false,
                                  debugContext()));
      }
      toReduceScatter = c;
    }
  }

  poplar::OptionFlags reduceScatterOptions = dv_p->lowering().gclOptions;
  reduceScatterOptions.set("useReplicatedImplementation", "true");

  poplar::Tensor reducedScattered = gcl::reduceScatterCrossReplica(
      graph().getPoplarGraph(),
      toReduceScatter.flatten().getPoplarTensor(),
      getPoplarCollectiveOperator(rrsOp.getCollectiveOp()),
      prog.getPoplarSequence(),
      toGCLCommGroup(rrsOp.getGCLCommGroup()),
      debugContext("replicatedReduceScatter"),
      reduceScatterOptions);

  setOutTensor(ReplicatedReduceScatterOp::getOutIndex(),
               snap::Tensor{reducedScattered, graph()});
}

InputCreatorType
ReplicatedReduceScatterOpx::getInputCreatorType(InIndex index) const {
  const auto &rrsOp = getOp<ReplicatedReduceScatterOp>();

  bool canCreate = false;

  if (hasInput(ReplicatedAllGatherOp::getCollectiveLinkedIndex())) {
    auto group = getCollectiveLinkedGroup();
    for (Op *cbrOp : group.second) {
      // Can't exist on itself
      if (cbrOp->id != rrsOp.id) {
        // This ReplicatedReduceScatterOp is not alone in a group, and can
        // use a pre-existing CBR to create the tensor layout
        // T34831 currently always disable creator, because it can lead to
        // rearrangements & padding in the created tensor that other consumers
        // may not be able to deal with
        canCreate = false;
      }
    }

    if (rrsOp.getCollectiveOp() == CollectiveOperator::Local) {
      // We currently have to disable canCreate for local reductions, because
      // it (with RTS) leads to circular dependencies where a weight's
      // layout can depend on itself, if the weight's other consumers
      // aren't higher-priority creators
      canCreate = false;
    }
  }

  return index == ReplicatedReduceScatterOp::getInIndex() && canCreate
             ? InputCreatorType::CanCreate
             : PopOpx::getInputCreatorType(index);
}

snap::Tensor ReplicatedReduceScatterOpx::createInputTensor(
    int inIndex,
    const poplar::DebugNameAndId &dnai) const {
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

  const auto &rrsOp = getOp<ReplicatedReduceScatterOp>();
  const auto &type  = popType(rrsOp.inTensor(inIndex)->info);
  auto input        = cbr->createCollectivesTensor(type, dnai.getPathName());
  return snap::Tensor{input, graph()};
}

DnfTensorIds
ReplicatedReduceScatterOpx::mustExistBeforeCreateDNF(InIndex) const {
  const auto &rrsOp = getOp<ReplicatedReduceScatterOp>();
  auto group        = getCollectiveLinkedGroup();
  DnfTensorIds mustExist;
  for (Op *cbrOp : group.second) {
    // Can't exist on itself
    if (cbrOp->id != rrsOp.id) {
      mustExist.push_back({cbrOp->inId(CollectivesBaseOp::getInIndex()),
                           cbrOp->outId(CollectivesBaseOp::getOutIndex())});
    }
  }

  logging::opx::trace(
      "ReplicatedReduceScatterOpx::mustExistBeforeCreateDNF, Op "
      "{}, must exist: {}",
      rrsOp.debugName(),
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
