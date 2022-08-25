// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <gcl/Collectives.hpp>
#include <snap/Graph.hpp>
#include <snap/Program.hpp>
#include <snap/Tensor.hpp>
#include <vector>
#include <poplar/Tensor.hpp>
#include <popops/DynamicSlice.hpp>
#include <popart/op/collectives/multi_replicatedallgather.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/popx/op/collectives/multi_replicatedallgatherx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/tensorindex.hpp>

#include "popart/commgroup.hpp"
#include "popart/graphcoreoperators.hpp"
#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/op/collectives/collectives.hpp"
#include "popart/popx/op/collectives/collectivesx.hpp"
#include "popart/popx/popopx.hpp"
#include "popart/region.hpp" // IWYU pragma: keep
#include "popart/util.hpp"

namespace popart {
namespace popx {

MultiReplicatedAllGatherOpx::MultiReplicatedAllGatherOpx(popart::Op *op,
                                                         Devicex *devicex)
    : MultiCollectiveBaseOpx(op, devicex) {
  verifyOp<MultiReplicatedAllGatherOp>(
      op, Onnx::CustomOperators::MultiReplicatedAllGather);
  inputCreatorPriority = -1.0;
}

void MultiReplicatedAllGatherOpx::growPart(OpxGrowPartId id) const {
  MultiReplicatedAllGatherOp &myOp = getOp<MultiReplicatedAllGatherOp>();
  MultiCollectivesOpxState *state =
      dv_p->lowering().getOpxState<MultiCollectivesOpxState>(myOp.id);
  OutIndex outputIndex = (OutIndex)id;

  // The input tensor is a pure pass-through
  poplar::Tensor inputTensor  = getInTensor(id).flatten().getPoplarTensor();
  state->configuredInputs[id] = inputTensor;

  // The output tensor should be dyamically sliceable
  size_t numSlices            = myOp.getCommSize();
  poplar::Tensor outputTensor = popops::createSliceableTensorFromSlice(
      outGraph(outputIndex).getPoplarGraph(),
      inputTensor.expand({0}),
      {0},
      {numSlices});
  state->configuredOutputs[id] = outputTensor;

  if (getOp<MultiReplicatedAllGatherOp>().undoRearrangeGrowPartForCollective(
          id)) {
    auto cbr = getCollectiveBalancedReorder(outputIndex);
    if (cbr) {
      outputTensor = cbr->undoRearrangeForCollective(outputTensor);
    } else {
      throw error(
          "MultiReplicatedAllGatherOpx::grow, CollectiveBalancedReorder "
          "not found for Op {}",
          op_p->debugName());
    }
  }
  setOutTensor(id,
               snap::Tensor(outputTensor.reshape(myOp.outInfo(id).shape_szt()),
                            outGraph(outputIndex)));
}

void MultiReplicatedAllGatherOpx::grow(snap::program::Sequence &prog) const {
  logging::opx::debug("Growing MultiReplicatedAllGatherOpx");
  auto &myOp = getOp<MultiReplicatedAllGatherOp>();
  MultiCollectivesOpxState *state =
      dv_p->lowering().getOpxState<MultiCollectivesOpxState>(myOp.id);

  // The inputs should be concatenated in order of ascending virtual graphId
  std::vector<InIndex> inputOrder;
  for (InIndex i{0}; i < myOp.output->n(); i++) {
    inputOrder.emplace_back(i);
  }

  auto vgid = [&myOp](InIndex in) {
    return myOp.getIntrospectionInVirtualGraphId(in).first;
  };

  std::stable_sort(
      inputOrder.begin(),
      inputOrder.end(),
      [&vgid](const auto a, const auto b) { return vgid(a) < vgid(b); });

  std::vector<poplar::Tensor> toGather;
  std::vector<poplar::Tensor> destinations;
  for (InIndex i : inputOrder) {
    toGather.emplace_back(state->configuredInputs[i]);
    destinations.emplace_back(state->configuredOutputs[i]);
  }

  // GCL to destination API requires a vector of tensors
  // we provide a vector of length 1
  std::vector<poplar::Tensor> toGatherTensor{poplar::concat(toGather, 0)};
  std::vector<poplar::Tensor> destinationTensor{
      poplar::concat(destinations, 1)};

  // Perform the GCL call on the concatenated tensors
  gcl::allGatherToDestinationCrossReplica(
      dv_p->lowering().graph().getPoplarGraph(),
      toGatherTensor,
      destinationTensor,
      prog.getPoplarSequence(),
      toGclCommGroup(myOp.getReplicaGrouping()),
      "MultiReplicatedAllGather",
      dv_p->lowering().gclOptions);
}

InputCreatorType
MultiReplicatedAllGatherOpx::getInputCreatorType(InIndex index) const {
  return index == 0 ? InputCreatorType::CanCreateOrUnwind
                    : PopOpx::getInputCreatorType(index);
}

snap::Tensor
MultiReplicatedAllGatherOpx::unwindTensorLayout(snap::Tensor tensor,
                                                InIndex in,
                                                OutIndex out) const {
  auto cbr = createCollectiveBalancedReorder(tensor, in);
  return snap::Tensor{cbr->createReplicaSlice(tensor.elementType()),
                      inGraph(in)};
}

view::RegMap MultiReplicatedAllGatherOpx::unwindRegion(InIndex in,
                                                       OutIndex out) const {
  auto info = inInfo(0);
  return [info](const view::Region &) {
    return view::Regions(1, view::Region::getFull(info.shape()));
  };
}

std::set<TensorId>
MultiReplicatedAllGatherOpx::mustExistBeforeCreate(InIndex) const {
  return {};
}

snap::Tensor MultiReplicatedAllGatherOpx::createInputTensor(
    InIndex index,
    const poplar::DebugNameAndId &dnai) const {
  auto &op = getOp<MultiReplicatedAllGatherOp>();

  if (index < op.output->n()) {
    auto outInfo = op.outInfo(index);
    auto outTensor =
        snap::Tensor{inGraph(index).getPoplarGraph().addVariable(
                         popType(outInfo), outInfo.shape_szt(), dnai),
                     inGraph(index)};
    dv_p->lowering().getLinearMapper().mapTensor(inGraph(index), outTensor);
    auto cbr = createCollectiveBalancedReorder(outTensor, index);
    return snap::Tensor{cbr->createReplicaSlice(popType(outInfo)),
                        inGraph(index)};
  }

  throw error("MultiReplicatedAllGatherOpx::createInput: Invalid index = " +
              std::to_string(index));
}

bool MultiReplicatedAllGatherOpx::hasCreatorViewChangers(InIndex index) const {
  auto &op = getOp<MultiReplicatedAllGatherOp>();
  return (index < op.output->n());
}

ViewChangers
MultiReplicatedAllGatherOpx::getCreatorViewChangers(InIndex index) const {
  auto &op = getOp<MultiReplicatedAllGatherOp>();
  if (index < op.input->n()) {
    ViewChangers viewChangers(
        {std::make_shared<ReplicatedGatherInScatterOutViewChanger>(
            inInfo(index).nelms(), getCollectiveLinkedGroup(index).id)});
    return viewChangers;
  }
  throw error(
      "MultiReplicatedAllGatherOpx::getCreatorViewChangers: Invalid index = " +
      std::to_string(index));
}

namespace {
OpxCreator<MultiReplicatedAllGatherOpx> MultiReplicatedAllGatherOpxCreator(
    Onnx::CustomOperators::MultiReplicatedAllGather);
}
} // namespace popx
} // namespace popart
