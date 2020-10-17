// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <gcl/Collectives.hpp>

#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/collectives/replicatedallgather.hpp>
#include <popart/popx/devicex.hpp>
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

  poplar::OptionFlags allGatherOptions = dv_p->gclOptions;
  allGatherOptions.set("useReplicatedImplementation", "true");

  poplar::Tensor gathered =
      gcl::allGather(graph(),
                     getInTensor(ReplicatedAllGatherOp::getInIndex()),
                     prog,
                     debugPrefix("replicatedAllGather"),
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
      gathered.reshape(
          op.outInfo(ReplicatedAllGatherOp::getOutIndex()).shape_szt()));
}

InputCreatorType
ReplicatedAllGatherOpx::getInputCreatorType(InIndex index) const {
  return index == ReplicatedAllGatherOp::getInIndex() &&
                 hasInput(ReplicatedAllGatherOp::getCollectiveLinkedIndex())
             ? InputCreatorType::CanCreateOrUnwind
             : Opx::getInputCreatorType(index);
}

poplar::Tensor ReplicatedAllGatherOpx::unwindTensorLayout(poplar::Tensor tensor,
                                                          InIndex,
                                                          OutIndex) const {
  auto replicationFactor = dv_p->getReplicationFactor();
  auto cbr               = createCollectiveBalancedReorder(tensor);
  auto rearranged        = cbr->rearrangeForCollective(tensor);
  // Rearranged tensor is always sliceable by the replication factor
  return rearranged.slice(0, rearranged.numElements() / replicationFactor);
}

view::RegMap ReplicatedAllGatherOpx::unwindRegion(InIndex, OutIndex) const {
  auto info = inInfo(ReplicatedAllGatherOp::getInIndex());
  return [info](const view::Region &) {
    return view::Regions(1, view::Region::getFull(info.shape()));
  };
}

std::vector<TensorId>
ReplicatedAllGatherOpx::mustExistBeforeCreate(InIndex) const {
  return {};
}

poplar::Tensor
ReplicatedAllGatherOpx::createInput(InIndex index,
                                    const std::string &name) const {
  auto &op = getOp<ReplicatedAllGatherOp>();

  if (index == ReplicatedAllGatherOp::getInIndex()) {
    auto outInfo = op.outInfo(ReplicatedAllGatherOp::getOutIndex());
    auto outTensor =
        graph().addVariable(popType(outInfo), outInfo.shape_szt(), name);
    dv_p->getLinearMapper().mapTensor(graph(), outTensor);
    auto inShape = op.inShape(ReplicatedAllGatherOp::getInIndex());

    return unwindTensorLayout(outTensor,
                              ReplicatedAllGatherOp::getInIndex(),
                              ReplicatedAllGatherOp::getOutIndex());
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
