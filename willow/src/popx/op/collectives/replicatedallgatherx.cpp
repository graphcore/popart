// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <gcl/Collectives.hpp>

#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/collectives/replicatedallgather.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/popx/op/collectives/replicatedallgatherx.hpp>
#include <popart/popx/opxmanager.hpp>

namespace popart {
namespace popx {

ReplicatedAllGatherOpx::ReplicatedAllGatherOpx(Op *op, Devicex *devicex)
    : CollectivesBaseOpx(op, devicex) {
  verifyOp<ReplicatedAllGatherOp>(op,
                                  Onnx::CustomOperators::ReplicatedAllGather);
  inputCreatorPriority = -1.0;
}

void ReplicatedAllGatherOpx::grow(poplar::program::Sequence &prog) const {
  auto &op = getOp<ReplicatedAllGatherOp>();

  poplar::OptionFlags allGatherOptions = dv_p->lowering().gclOptions;
  allGatherOptions.set("useReplicatedImplementation", "true");

  poplar::Tensor gathered = gcl::allGather(
      graph().getPoplarGraph(),
      getInTensor(ReplicatedAllGatherOp::getInIndex()).getPoplarTensor(),
      prog,
      debugContext("replicatedAllGather"),
      allGatherOptions);
  if (hasInput(ReplicatedAllGatherOp::getCollectiveLinkedIndex())) {
    auto cbr = getCollectiveBalancedReorder();
    if (cbr) {
      gathered = cbr->undoRearrangeForCollective(gathered);
    } else {
      throw error("ReplicatedAllGatherOpx::grow, "
                  "CollectiveBalancedReorder not found for Op {}",
                  op_p->debugName());
    }
  }

  setOutTensor(
      ReplicatedAllGatherOp::getOutIndex(),
      snap::Tensor{
          gathered.reshape(
              op.outInfo(ReplicatedAllGatherOp::getOutIndex()).shape_szt()),
          graph()});
}

InputCreatorType
ReplicatedAllGatherOpx::getInputCreatorType(InIndex index) const {
  return index == ReplicatedAllGatherOp::getInIndex() &&
                 hasInput(ReplicatedAllGatherOp::getCollectiveLinkedIndex())
             ? InputCreatorType::CanCreateOrUnwind
             : PopOpx::getInputCreatorType(index);
}

snap::Tensor ReplicatedAllGatherOpx::unwindTensorLayout(snap::Tensor tensor,
                                                        InIndex,
                                                        OutIndex) const {
  auto cbr = createCollectiveBalancedReorder(tensor);
  return snap::Tensor{
      cbr->createReplicaSlice(tensor.getPoplarTensor().elementType()), graph()};
}

view::RegMap ReplicatedAllGatherOpx::unwindRegion(InIndex, OutIndex) const {
  auto info = inInfo(ReplicatedAllGatherOp::getInIndex());
  return [info](const view::Region &) {
    return view::Regions(1, view::Region::getFull(info.shape()));
  };
}

std::set<TensorId>
ReplicatedAllGatherOpx::mustExistBeforeCreate(InIndex) const {
  return {};
}

snap::Tensor ReplicatedAllGatherOpx::createInputTensor(
    InIndex index,
    const poplar::DebugNameAndId &dnai) const {
  auto &op = getOp<ReplicatedAllGatherOp>();

  if (index == ReplicatedAllGatherOp::getInIndex()) {
    auto outInfo = op.outInfo(ReplicatedAllGatherOp::getOutIndex());
    auto outTensor =
        snap::Tensor{graph().getPoplarGraph().addVariable(
                         popType(outInfo), outInfo.shape_szt(), dnai),
                     graph()};
    dv_p->lowering().getLinearMapper().mapTensor(graph(), outTensor);
    auto cbr = createCollectiveBalancedReorder(outTensor);
    return snap::Tensor{cbr->createReplicaSlice(popType(outInfo)), graph()};
  }

  throw error("ReplicatedAllGatherOpx::createInput: Invalid index = " +
              std::to_string(index));
}

bool ReplicatedAllGatherOpx::hasCreatorViewChangers(InIndex index) const {
  return (index == ReplicatedAllGatherOp::getInIndex());
}

ViewChangers
ReplicatedAllGatherOpx::getCreatorViewChangers(InIndex index) const {
  if (index == ReplicatedAllGatherOp::getInIndex()) {
    ViewChangers viewChangers(
        {std::make_shared<ReplicatedGatherInScatterOutViewChanger>(
            inInfo(ReplicatedAllGatherOp::getInIndex()).nelms(),
            getCollectiveLinkedGroup().first)});
    return viewChangers;
  }
  throw error(
      "ReplicatedAllGatherOpx::getCreatorViewChangers: Invalid index = " +
      std::to_string(index));
}

namespace {
OpxCreator<ReplicatedAllGatherOpx>
    ReplicatedAllGatherOpxCreator(Onnx::CustomOperators::ReplicatedAllGather);
} // namespace
} // namespace popx
} // namespace popart
