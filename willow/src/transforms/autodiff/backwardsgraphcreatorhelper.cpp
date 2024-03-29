// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <boost/range/algorithm/find.hpp>
#include <graphfromlosstolossupdater.hpp>
#include <memory>
#include <numeric>
#include <set>
#include <transforms/autodiff/autodiffiradapter.hpp>
#include <transforms/autodiff/backwardsgraphcreatorhelper.hpp>
#include <transforms/autodiff/gradgrower.hpp>
#include <transforms/autodiff/gradgrowerop.hpp>
#include <transforms/autodiff/gradgrowersumop.hpp>
#include <utility>
#include <popart/alias/aliasmodel.hpp>
#include <popart/alias/aliasmodelgrower.hpp>
#include <popart/bwdgraphinfo.hpp>
#include <popart/graph.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>
#include <popart/tensornames.hpp>
#include <popart/util.hpp>

#include "popart/error.hpp"
#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/tensordata.hpp"
#include "popart/tensors.hpp"
#include "popart/vertex.hpp"

namespace popart {

BackwardsGraphCreatorHelper::BackwardsGraphCreatorHelper(const Graph &fwdGraph_,
                                                         Graph &bwdGraph_)
    : fwdGraph(fwdGraph_), bwdGraph(bwdGraph_) {}

BwdGraphInfo BackwardsGraphCreatorHelper::populateBwdGraph(
    const nonstd::optional<TensorIds> &gradsProvidedForFwdId,
    const nonstd::optional<TensorIds> &gradsRequiredForFwdId,
    const FwdGraphToBwdGraphInfo &calledGraphsGradInfo) {

  // Set `gradsProvided` to the list of available gradients of ouputs of the
  // forward graph. If `gradsProvidedForFwdId` is explicitly provided, use it.
  // If not, add each output of the forward graph and we will prune
  // unneeded ones in the `doPrune` call.
  auto gradsProvided = gradsProvidedForFwdId.has_value()
                           ? *gradsProvidedForFwdId
                           : fwdGraph.getOutputIds();

  growGradGraph(gradsProvided, gradsRequiredForFwdId, calledGraphsGradInfo);

  // Remove fwd ops from bwdGraph that we don't need. Do not allow removing
  // explicitly provided inputs from gradsProvided.
  std::vector<InIndex> inputIndices;
  inputIndices.resize(gradsProvided.size());
  std::iota(inputIndices.begin(), inputIndices.end(), 0);

  if (gradsProvidedForFwdId.has_value()) {
    // Do not remove explicitly provided inputs, warn instead.
    doPrune(
        bwdGraph, inputIndices, WarnIfProtectedInputCouldHaveBeenRemoved::Yes);
  } else {
    // Remove whatever inputs we want, don't warn.
    doPrune(bwdGraph, {}, WarnIfProtectedInputCouldHaveBeenRemoved::No);
  }

  // It is guaranteed that provided inputs are expected inputs
  // connections. Pass them to make grad info.
  std::map<InIndex, ExpectedConnection> expectedConnectionsMap;
  if (gradsProvidedForFwdId.has_value()) {
    auto gradsProvided = *gradsProvidedForFwdId;
    int numProvided    = gradsProvided.size();
    for (int i = 0; i < numProvided; i++) {
      ExpectedConnection expectedConnection;
      expectedConnection.fwdId = gradsProvided[i];
      expectedConnection.type  = ExpectedConnectionType::Fwd;
      expectedConnectionsMap.insert({i, expectedConnection});
    }
  }

  return makeGradInfo(expectedConnectionsMap);
}

BwdGraphInfo BackwardsGraphCreatorHelper::makeGradInfo(
    std::map<InIndex, ExpectedConnection> &expectedConnectionsMap) {

  // NOTE: Code later on in the stack (getGradOps for subgraph ops) has certain
  // requirements on what expected inputs/outputs can be. Namely, expected
  // inputs can be non-grad of an input of fwdGraph or a non-grad or grad of
  // of an output of fwdGraph. Expected outputs must be grads of fwdGraph
  // inputs. This is currently true by construction but `makeGradInfo` is able
  // to deal with the general case. If these requirements are broken by a future
  // version of BackwardsGraphCreatorHelper then it is likely that this
  // results in an error in, say, `OpWithCalledGraphs` code.

  auto populateExpConns =
      [this, &expectedConnectionsMap](ExpectedConnections &expConns,
                                      const std::vector<TensorId> &bwdIds,
                                      bool isInputs) {
        for (InIndex i = 0; i < bwdIds.size(); i++) {
          auto bwdId = bwdIds.at(i);
          if (bwdIdIsNonGrad(bwdId)) {
            // Non-grad tensor in bwdGraph.
            auto fwdId = bwdNonGradIdToFwdId(bwdId);
            expConns.push_back({fwdId, ExpectedConnectionType::Fwd});
          } else {
            // Grad tensor in bwdGraph.
            // Populate expected connections by bwdGradIdToFwdId. But employ
            // expectedConnectionsMap if it is given - to deal with
            // subgraph provided inputs in growGradSumOp.
            TensorId fwdId = (isInputs && expectedConnectionsMap.count(i))
                                 ? expectedConnectionsMap[i].fwdId
                                 : bwdGradIdToFwdId(bwdId);
            expConns.push_back({fwdId, ExpectedConnectionType::FwdGrad});
          }
        }
      };

  ExpectedConnections expectedInputs;
  ExpectedConnections expectedOutputs;

  populateExpConns(expectedInputs, bwdGraph.getInputIds(), true);
  // IMPORTANT NOTE:
  // It is a requirement of autodiff that the returned ExpectedConnections for
  // the outputs of the bwdGraph be in order of gradsRequiredForFwdId (passed to
  // populateBwdGraph).
  //
  // That method will create the bwdGraph outputs in the order of
  // gradsRequiredForFwdId. This loop then adds the ExpectedConnections in order
  // of those outputs, thus abiding by our requirement.
  //
  // Therefore, THE ORDER OF THE OUTPUT IDS PASSED HERE IS VERY IMPORTANT.
  populateExpConns(expectedOutputs, bwdGraph.getOutputIds(), false);

  return BwdGraphInfo{bwdGraph.id, expectedInputs, expectedOutputs};
}

void BackwardsGraphCreatorHelper::growGradGraph(
    const TensorIds &gradsProvidedForFwdId,
    const nonstd::optional<TensorIds> &gradsRequiredForFwdId,
    const FwdGraphToBwdGraphInfo &calledGraphsGradInfo) {
  // clone ops from the fwdGraph into the bwdGraph

  // Clone all tensors in fwdGraph as tensors in bwdGraph and keep track in
  // fwdToBwdTensorIdMap for later processing. Mapping key is TensorId rather
  // than Tensor* to make ordering independent of memory allocation.
  std::map<TensorId, TensorId> fwdToBwdTensorIdMap;
  for (auto &fwdId : fwdGraph.getTensors().getAllTensorIds()) {
    auto fwdTensor = fwdGraph.getTensors().get(fwdId);
    auto bwdId     = addScope(bwdGraph, removeScope(fwdGraph, fwdId));
    auto bwdTensor = fwdTensor->clone(bwdGraph);
    bwdTensor->id  = bwdId;
    if (fwdTensor->hasTensorData()) {
      bwdTensor->setTensorDataFromCopyOf(fwdTensor->tensorData()->data(),
                                         fwdTensor->tensorData()->size());
    }
    fwdToBwdTensorIdMap[fwdTensor->id] = bwdTensor->id;
    bwdGraph.getTensors().moveIntoTensors(std::move(bwdTensor));
  }

  std::vector<GradNonGradTensor> gradNonGradTensors;
  gradNonGradTensors.reserve(gradsProvidedForFwdId.size());

  // TODO(T56304): Remove need to cast away const-ness of fwdGraph. Currently,
  // we have to set the following state:
  //   - fromLoss/toLoss, which we unset immediately after anyway.
  //   - op->setCalledSubgraphGradInfo
  auto &ncFwdGraph = const_cast<Graph &>(fwdGraph);

  // Initialise FromLoss::No for all forward ops
  for (auto &id_op : ncFwdGraph.getOps()) {
    auto op      = id_op.second.get();
    op->fromLoss = PathFromLoss::No;
  }

  for (auto t : ncFwdGraph.getTensors().getAll()) {
    if (t->hasProducer()) {
      auto op     = t->getProducerUnsafe();
      t->toLoss   = op->toLoss;
      t->fromLoss = op->fromLoss;
    }
  }

  // Create a gradient input tensor for each output tensor of fwdGraph
  for (auto &gradProvided : gradsProvidedForFwdId) {
    if (!fwdGraph.hasOutputId(gradProvided)) {
      logging::warn("[Autodiff] Did not expect the gradient of '{}' to be "
                    "provided to the backward graph of {} (it is not an "
                    "output of this graph)",
                    gradProvided,
                    ncFwdGraph.getGraphString());
    }
    auto gradId   = fwdIdToBwdGradId(gradProvided);
    auto gradInfo = fwdGraph.getTensors().get(gradProvided)->info;
    bwdGraph.addInput(gradId, gradInfo);
    gradTensorMap.insert({gradProvided, gradId});

    gradNonGradTensors.push_back({bwdGraph.getTensors().get(gradId),
                                  fwdGraph.getTensors().get(gradProvided)});

    // Initialise FromLoss and ToLoss to ::Yes for all "losses"
    auto t      = ncFwdGraph.getTensors().get(gradProvided);
    t->fromLoss = PathFromLoss::Yes;
    t->toLoss   = PathToLoss::Yes;
    if (t->hasProducer()) {
      auto op      = t->getProducerUnsafe();
      op->toLoss   = PathToLoss::Yes;
      op->fromLoss = PathFromLoss::No;
    }
  }

  // Needed for GradGrower. We reset all vertices to ::Undefined after, as the
  // setting is only valid on main graph vertices.
  graphFromLossToLossUpdater::propagate(ncFwdGraph);

  AliasModel bwdGraphAliasModel;
  AliasModelGrower aliasModelGrower{bwdGraphAliasModel};
  aliasModelGrower.growFullGraph(bwdGraph, DataDependenciesOnly::Yes);

  AutodiffIrAdapter adapter{bwdGraph.getIr()};
  GradGrowerOp gog{adapter};
  GradGrowerSumOp gsg{adapter};

  GradGrower ad{ncFwdGraph};
  ad.growGrads(bwdGraph,
               gradNonGradTensors,
               {},
               gog,
               gsg,
               const_cast<FwdGraphToBwdGraphInfo &>(calledGraphsGradInfo),
               bwdGraphAliasModel);

  // Reset all vertices to have from/to loss ::Undefined.
  graphFromLossToLossUpdater::unsetAll(ncFwdGraph);

  /* Connect up outputs */

  // If null gradsRequiredFor, we output as many input grads as possible.
  if (!gradsRequiredForFwdId.has_value()) {
    for (auto &fwdId : fwdGraph.getInputIds()) {
      // Try to find grad tensor of fwd tensor in bwd graph. If it exists, add
      // the mapping to `gradTensorMap` and mark the tensor as an output. Note
      // `gradTensorMap` is a class member used outside this function too, so we
      // need to preserve that this side-effect has happened by the end of this
      // function.

      auto gradId = fwdIdToBwdGradId(fwdId);
      if (bwdGraph.getTensors().contains(gradId)) {
        gradTensorMap.insert({fwdId, gradId});
        bwdGraph.markAsOutput(gradId);
      }
    }
    // Else, we try to find all the requested grads and output them. If one is
    // not found, it is an error. Recall, these can be any grad tensors, not
    // just grads of the fwd input tensors. It is valid in Popart for a tensor
    // to be an output and also consumed by further ops in the subgraph.
  } else {
    // IMPORTANT NOTE:
    // It is a requirement of autodiff that the returned ExpectedConnections for
    // the outputs of the bwdGraph be in order of gradsRequiredForFwdId.
    //
    // This loop is adding bwdGraph outputs in the order of
    // gradsRequiredForFwdId. Then, later, makeGradInfo will generate the
    // ExpectedConnections in order of the bwdGraph outputs.
    //
    // Therefore, THE ORDER OF THIS LOOP IS VERY IMPORTANT.
    for (const auto &fwdId : *gradsRequiredForFwdId) {
      auto gradId = fwdIdToBwdGradId(fwdId);
      if (bwdGraph.getTensors().contains(gradId)) {
        gradTensorMap.insert({fwdId, gradId});
        bwdGraph.markAsOutput(gradId);
      } else {
        throw error("[Autodiff] Unable to provide required gradient output for "
                    "fwd id '{}' "
                    "in bwd graph {}",
                    fwdId,
                    bwdGraph.getGraphString());
      }
    }
  }

  // Go over cloned forward tensors in fwdToBwdTensorIdMap to see if they are
  // actually being used. Any actually used, mark them as input. Any unused,
  // remove them.
  for (const auto &entry : fwdToBwdTensorIdMap) {
    const auto &bwdId = entry.second;
    auto bwdTensor    = bwdGraph.getTensors().get(bwdId);

    if (bwdTensor->consumers.getTotal() <= 0) {
      // If the tensor is not consumed by any grad ops then it's not really
      // needed in bwdGraph. Let's remove it.
      bwdGraph.getTensors().remove(bwdId);
      bwdTensor = nullptr;
    } else {
      // If the tensor is used, mark it as an input.
      bwdGraph.markAsInput(bwdId);
    }
  }
}

std::vector<OpId> Graph::getOpIds() const {
  const auto &opMap = getOps();
  std::vector<OpId> opIds;
  opIds.reserve(opMap.size());
  for (const auto &x : opMap) {
    opIds.push_back(x.first);
  }
  return opIds;
}

bool BackwardsGraphCreatorHelper::bwdIdIsNonGrad(const TensorId &id) {
  auto x = ::popart::removeScope(bwdGraph, id);
  return !::popart::isGradId(x);
}

TensorId BackwardsGraphCreatorHelper::fwdIdToBwdGradId(const TensorId &id) {
  return ::popart::fwdIdToBwdGradId(fwdGraph, bwdGraph, id);
}

TensorId BackwardsGraphCreatorHelper::bwdGradIdToFwdId(const TensorId &id) {
  return ::popart::bwdGradIdToFwdId(fwdGraph, bwdGraph, id);
}

TensorId BackwardsGraphCreatorHelper::bwdNonGradIdToFwdId(const TensorId &id) {
  return ::popart::bwdNonGradIdToFwdId(fwdGraph, bwdGraph, id);
}

TensorId BackwardsGraphCreatorHelper::getInputTensorId(
    Op *nonGradOp,
    const GradInOutMapper &inOutMapper) {
  TensorId result;

  int indexFwd      = inOutMapper.iNonGrad;
  GradOpInType type = inOutMapper.type;

  // the input at index 'indexGrad' to gradOp is
  switch (type) {
  //  (1) the INPUT at index 'indexFwd' of nonGradOp
  //  This will be a tensor internal to fwdGraph
  case GradOpInType::In: {
    auto fwdId = nonGradOp->inId(indexFwd);
    auto bwdId = addScope(bwdGraph, removeScope(fwdGraph, fwdId));
    return bwdId;
  }

  //  (2) the OUTPUT at index 'indexFwd' of nonGradOp
  //  This will be a tensor internal to fwdGraph
  case GradOpInType::Out: {
    auto fwdId = nonGradOp->outId(indexFwd);
    auto bwdId = addScope(bwdGraph, removeScope(fwdGraph, fwdId));
    return bwdId;
  }

  //  (3) the GRADIENT of the OUTPUT
  //      at index 'indexFwd' of nonGradOp.
  case GradOpInType::GradOut: {
    auto fwdId = nonGradOp->outId(indexFwd);
    auto found = gradTensorMap.find(fwdId);
    if (found == gradTensorMap.end()) {
      throw error("Could not find TensorId '{}' in gradTensorMap", fwdId);
    }
    return found->second;
  }

  default: {
    throw internal_error("Unsupported value for GradOpInType");
  }
  }
}
void BackwardsGraphCreatorHelper::doPrune(
    Graph &graph,
    const std::vector<InIndex> &protectedInputIndices,
    WarnIfProtectedInputCouldHaveBeenRemoved warn) {
  using boost::range::find;

  // Get output ids.
  const auto outputIds = graph.getOutputIds();
  auto isNotOutput     = [&outputIds](const TensorId &tId) {
    return find(outputIds, tId) == outputIds.end();
  };

  // Get protected input ids from input indices. Put them in a set.
  std::set<TensorId> protectedInputIds;
  for (InIndex index : protectedInputIndices) {
    protectedInputIds.insert(graph.getInputId(index));
  }

  // Lambda for checking if an id is protected.
  auto isProtectedInputId = [&](const TensorId &tId) {
    return protectedInputIds.find(tId) != protectedInputIds.end();
  };

  // As the below is a fixed point computation, we are at risk of the same code
  // emitting the same warning more than once. To avoid this, we keep track of
  // which tensors we've given warnings for already.
  std::set<TensorId> warningCache;
  auto processProtectedInputId = [&](const TensorId &id) {
    if (warn == WarnIfProtectedInputCouldHaveBeenRemoved::Yes) {
      if (warningCache.find(id) == warningCache.end()) {
        logging::warn("[Autodiff] Input '{}' of the backwards graph {} is "
                      "not used but cannot be pruned because it was "
                      "explicitly provided as an input (and removing it "
                      "would break an autodiff post-condition).",
                      id,
                      graph.getGraphString());
        warningCache.insert(id);
      }
    }
  };

  while (true) {
    // Set to true if a tensor or op is removed.
    bool continueLoop = false;

    auto &tensors = graph.getTensors();

    // Remove tensors that are not inputs or outputs and have no consumers.
    const auto tensorIds = tensors.getAllTensorIds();
    for (auto id : tensorIds) {
      auto tensor = graph.getTensors().get(id);
      if (tensor->consumers.getTotal() == 0 && isNotOutput(id)) {
        if (isProtectedInputId(id)) {
          processProtectedInputId(id);
        } else {
          bool canRemove = true;
          if (tensor->hasProducer()) {
            // We can only remove a tensor that is produced if no outputs
            // of the producer are consumed.
            auto producer = tensor->getProducer();
            for (auto outTensor : producer->output->tensors()) {
              if (outTensor->consumers.getTotal() > 0 ||
                  !isNotOutput(outTensor->id)) {
                canRemove = false;
              }
            }
            if (canRemove) {
              producer->disconnectAllOutputs();
            }
          }

          if (canRemove) {
            tensors.remove(id);
            continueLoop = true;
          }
        }
      }
    }

    // Remove ops with no outputs.
    const auto opIds = graph.getOpIds();
    for (const auto id : opIds) {
      const auto op = graph.getOp(id);
      if (op->output->n() == 0) {
        op->disconnectAllInputs();
        graph.eraseOp(id);
        continueLoop = true;
      }
    }

    if (!continueLoop) {
      break;
    }
  }

  // Remove inputs ids that have been pruned.
  auto inputIds = graph.getInputIds();
  for (auto &id : inputIds) {
    if (!graph.getTensors().contains(id)) {
      auto inputIter = find(graph.getInputIds(), id);
      graph.graph_inputs.erase(inputIter);
    }
  }
}

} // namespace popart
