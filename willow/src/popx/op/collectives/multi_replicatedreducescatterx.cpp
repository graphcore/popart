// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <gcl/CollectiveBalancedReorder.hpp>
#include <gcl/Collectives.hpp>
#include <map>
#include <memory>
#include <set>
#include <stddef.h>
#include <string>
#include <utility>
#include <vector>
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <poplar/Tensor.hpp>
#include <popops/Zero.hpp>
#include <popart/op/collectives/multi_replicatedreducescatter.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/popx/op/collectives/multi_replicatedreducescatterx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/tensorindex.hpp>

#include "popart/error.hpp"
#include "popart/graphcoreoperators.hpp"
#include "popart/ir.hpp"
#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/op/collectives/collectives.hpp"
#include "popart/popx/debugcontextx.hpp"
#include "popart/popx/op/collectives/collectivesx.hpp"
#include "popart/popx/opx.hpp"
#include "popart/popx/viewchangers.hpp"
#include "popart/region.hpp" // IWYU pragma: keep
#include "popart/replicatedtensorsharding.hpp"
#include "popart/tensor.hpp"
#include "popart/tensorinfo.hpp"
#include "popart/tensorlocation.hpp"
#include "popart/util.hpp"

namespace popart {
namespace popx {

MultiReplicatedReduceScatterOpx::MultiReplicatedReduceScatterOpx(
    popart::Op *op,
    Devicex *devicex)
    : MultiCollectiveBaseOpx(op, devicex) {
  verifyOp<MultiReplicatedReduceScatterOp>(
      op, Onnx::CustomOperators::MultiReplicatedReduceScatter);
}

InputCreatorType
MultiReplicatedReduceScatterOpx::getInputCreatorType(InIndex index) const {
  const auto &rrsOp = getOp<MultiReplicatedReduceScatterOp>();
  if (rrsOp.isCollectiveLinkedIndexTensor(index)) {
    return Opx::getInputCreatorType(index);
  }

  bool canCreate = false;
  auto group     = getCollectiveLinkedGroup(index);
  for (auto cbrOpId : group.collectiveOpIds) {
    // Can't exist on itself
    if (cbrOpId.first != rrsOp.id) {
      // This Op is not alone in a group, and can
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
  // inequality targets input tensors (other inputs are indices)
  return canCreate ? InputCreatorType::CanCreate
                   : Opx::getInputCreatorType(index);
}

poplar::Tensor MultiReplicatedReduceScatterOpx::createInput(
    InIndex idx,
    const poplar::DebugNameAndId &dnai) const {
  const auto &rrsOp = getOp<MultiReplicatedReduceScatterOp>();
  if (idx >= rrsOp.output->n()) {
    throw error("MultiReplicatedReduceScatterOpx::createInput, cannot create "
                "input at {}",
                idx);
  }

  auto cbr = getCollectiveBalancedReorder(idx);
  if (!cbr) {
    throw error("MultiReplicatedReduceScatterOpx::createInput, "
                "CollectiveBalancedReorder not found for Op {}",
                op_p->debugName());
  }

  const auto &type = popType(rrsOp.inTensor(idx)->info);
  auto input       = cbr->createCollectivesTensor(type, dnai.getPathName());
  return input;
}

// Prepare the output tensors
void MultiReplicatedReduceScatterOpx::growPart(OpxGrowPartId id) const {
  MultiReplicatedReduceScatterOp &myOp =
      getOp<MultiReplicatedReduceScatterOp>();
  MultiCollectivesOpxState *state =
      dv_p->lowering().getOpxState<MultiCollectivesOpxState>(myOp.id);
  poplar::Tensor toReduceScatter = getInTensor(id);
  InIndex inputIndex             = (InIndex)id;
  InIndex outputIndex            = (OutIndex)id;
  ReplicatedTensorShardingIndicesIndex groupIndex =
      (ReplicatedTensorShardingIndicesIndex)id;

  // May need to configure the input for replicated tensor sharding
  poplar::program::Sequence &progs = state->inputConfiguringPrograms;
  if (myOp.rearrangeGrowPartForCollective(id)) {
    auto group = getCollectiveLinkedGroup(inputIndex);

    ViewChangers viewChangers(
        {std::make_shared<ReplicatedGatherInScatterOutViewChanger>(
            outInfo(outputIndex).nelms(), group.id)});
    setOutViewChangers(outputIndex, viewChangers);

    if (!hasInViewChangers(inputIndex) ||
        getInViewChangers(inputIndex) != viewChangers) {
      logging::opx::debug(
          "MultiReplicatedReduceScatterOpx::growPart rearranging {}",
          inId(inputIndex));

      // Tensor not rearranged for reduceScatter yet, do it now
      auto cbr = createCollectiveBalancedReorder(toReduceScatter, groupIndex);
      auto cbrTensor = cbr->createCollectivesTensor(
          toReduceScatter.elementType(), inId(inputIndex));
      popops::zero(inGraph(inputIndex), cbrTensor, progs, debugContext());
      auto ref = cbr->undoRearrangeForCollective(cbrTensor);
      if (hasInViewChangers(inputIndex)) {
        progs.add(poplar::program::Copy(
            getInViewChangers(inputIndex).apply(toReduceScatter).flatten(),
            ref.flatten(),
            false,
            debugContext()));
      } else {
        progs.add(poplar::program::Copy(
            toReduceScatter.flatten(), ref.flatten(), false, debugContext()));
      }
      toReduceScatter = cbrTensor;
    }
  }

  state->configuredInputs[id] = toReduceScatter;

  // The output tensor should follow the layout of the input tensor
  size_t groupSize     = myOp.getCommSize();
  size_t numInElements = toReduceScatter.numElements();
  size_t numOutElems   = numInElements / groupSize;
  auto outputTensor    = toReduceScatter.flatten().slice(0, numOutElems, 0);
  outputTensor         = outGraph(outputIndex).clone(outputTensor);
  state->configuredOutputs[id] = outputTensor;
  setOutTensor(outputIndex, outputTensor);
}

void MultiReplicatedReduceScatterOpx::grow(
    poplar::program::Sequence &prog) const {
  MultiReplicatedReduceScatterOp &myOp =
      getOp<MultiReplicatedReduceScatterOp>();
  logging::opx::debug("[MultiReplicatedReduceScatterOpx] Growing  {}",
                      myOp.debugName());
  // The opx state populated in growPart
  MultiCollectivesOpxState *state =
      dv_p->lowering().getOpxState<MultiCollectivesOpxState>(myOp.id);

  // The input tensors should be concatenated such that the parts belonging
  // to the same replica are placed together and they should be sorted in order
  // of increasing virtual graph id (because of how the tensor will be handled
  // by GCL internaly)
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

  std::vector<poplar::Tensor> toReduceScatter;
  std::vector<poplar::Tensor> destinations;
  for (InIndex i : inputOrder) {
    poplar::Tensor flatInput         = state->configuredInputs[i];
    poplar::Tensor flatDestination   = state->configuredOutputs[i];
    unsigned int outputSize          = flatDestination.numElements();
    long unsigned int innerDimension = flatInput.numElements() / outputSize;
    long unsigned int outerDimension = flatInput.numElements() / innerDimension;
    toReduceScatter.emplace_back(
        flatInput.reshape({innerDimension, outerDimension}));
    destinations.emplace_back(flatDestination);
  }

  // There may be additional programs required to configure each one of these
  // inputs.
  prog.add(state->inputConfiguringPrograms);

  std::vector<poplar::Tensor> toReduceScatterTensor{
      poplar::concat(toReduceScatter, 1).flatten()};
  std::vector<poplar::Tensor> destinationTensor{
      poplar::concat(destinations, 0)};

  gcl::reduceScatterToDestinationCrossReplica(
      dv_p->lowering().graph(),
      toReduceScatterTensor,
      destinationTensor,
      getPoplarCollectiveOperator(myOp.getCollectiveOp()),
      prog,
      toGclCommGroup(myOp.getReplicaGrouping()),
      "MultiReplicatedReduceScatter",
      dv_p->lowering().gclOptions);
}

DnfTensorIds
MultiReplicatedReduceScatterOpx::mustExistBeforeCreateDNF(InIndex in) const {
  const auto &rrsOp = getOp<MultiReplicatedReduceScatterOp>();
  auto group        = getCollectiveLinkedGroup(in);
  DnfTensorIds mustExist;
  for (auto cbrOpId : group.collectiveOpIds) {
    // Can't depend on itself
    if (cbrOpId.first != rrsOp.id) {
      auto cbrOp = dv_p->ir().getOp(cbrOpId.first);
      mustExist.push_back({cbrOp->inId(in), cbrOp->outId(in)});
    }
  }

  logging::opx::trace(
      "MultiReplicatedReduceScatterOpx::mustExistBeforeCreateDNF, Op "
      "{}, must exist: {}",
      rrsOp.debugName(),
      mustExist);

  return mustExist;
}

ViewChangers
MultiReplicatedReduceScatterOpx::getCreatorViewChangers(InIndex index) const {
  MultiReplicatedReduceScatterOp &myOp =
      getOp<MultiReplicatedReduceScatterOp>();
  if (index < myOp.output->n()) {
    auto cbr = getCollectiveBalancedReorder(index);
    ViewChangers viewChangers(
        {std::make_shared<ReplicatedGatherOutScatterInViewChanger>(
            cbr, getCollectiveLinkedGroup(index).id)});
    return viewChangers;
  }
  throw error("MultiReplicatedReduceScatterOpx::getCreatorViewChangers: "
              "Invalid index = " +
              std::to_string(index));
}

namespace {
OpxCreator<MultiReplicatedReduceScatterOpx>
    MultiReplicatedReduceScatterOpxCreator(
        Onnx::CustomOperators::MultiReplicatedReduceScatter);
}
} // namespace popx
} // namespace popart
