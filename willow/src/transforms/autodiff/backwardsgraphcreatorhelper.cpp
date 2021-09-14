// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <transforms/autodiff/backwardsgraphcreatorhelper.hpp>

#include <popart/bwdgraphinfo.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/sum.hpp>
#include <popart/pbwrap.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>
#include <popart/tensornames.hpp>

#include <transforms/autodiff/gradgrowersumop.hpp>

#include <boost/range/algorithm/find.hpp>

namespace popart {

BackwardsGraphCreatorHelper::BackwardsGraphCreatorHelper(const Graph &fwdGraph_,
                                                         Graph &bwdGraph_)
    : fwdGraph(fwdGraph_), bwdGraph(bwdGraph_), gradOpStore() {}

BwdGraphInfo BackwardsGraphCreatorHelper::populateBwdGraph(
    const TensorIds &gradsProvidedForFwdId,
    const nonstd::optional<TensorIds> &gradsRequiredForFwdId,
    const FwdGraphToBwdGraphInfo &calledGraphsGradInfo) {

  growGradGraph(
      gradsProvidedForFwdId, gradsRequiredForFwdId, calledGraphsGradInfo);

  // Remove fwd ops from bwdGraph that we don't need.
  doPrune(bwdGraph);

  return makeGradInfo();
}

BwdGraphInfo BackwardsGraphCreatorHelper::makeGradInfo() {

  // NOTE: Code later on in the stack (getGradOps for subgraph ops) has certain
  // requirements on what expected inputs/outputs can be. Namely, expected
  // inputs can be non-grad of an input of fwdGraph or a non-grad or grad of
  // of an output of fwdGraph. Expected outputs must be grads of fwdGraph
  // inputs. This is currently true by construction but `makeGradInfo` is able
  // to deal with the general case. If these requirements are broken by a future
  // version of BackwardsGraphCreatorHelper then it is likely that this
  // results in an error in, say, `OpWithCalledGraphs` code.

  auto populateExpConns = [this](ExpectedConnections &expConns,
                                 const std::vector<TensorId> &bwdIds) {
    for (InIndex i = 0; i < bwdIds.size(); i++) {
      auto bwdId = bwdIds.at(i);
      if (bwdIdIsNonGrad(bwdId)) {
        // Non-grad tensor in bwdGraph.
        auto fwdId = bwdNonGradIdToFwdId(bwdId);
        expConns.push_back({fwdId, ExpectedConnectionType::Fwd});
      } else {
        // Grad tensor in bwdGraph.
        auto fwdId = bwdGradIdToFwdId(bwdId);
        expConns.push_back({fwdId, ExpectedConnectionType::FwdGrad});
      }
    }
  };

  ExpectedConnections expectedInputs;
  ExpectedConnections expectedOutputs;

  populateExpConns(expectedInputs, bwdGraph.getInputIds());
  populateExpConns(expectedOutputs, bwdGraph.getOutputIds());

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
    auto bwdId     = bwdGraph.addScope(fwdGraph.removeScope(fwdId));
    auto bwdTensor = fwdTensor->clone(bwdGraph);
    bwdTensor->id  = bwdId;
    if (fwdTensor->hasTensorData()) {
      bwdTensor->setTensorData(fwdTensor->info,
                               fwdTensor->tensorData()->data());
    }
    fwdToBwdTensorIdMap[fwdTensor->id] = bwdTensor->id;
    bwdGraph.getTensors().moveIntoTensors(std::move(bwdTensor));
  }

  // Create a gradient input tensor for each output tensor of fwdGraph
  for (auto &scopedId : fwdGraph.getOutputIds()) {
    if (std::find(gradsProvidedForFwdId.begin(),
                  gradsProvidedForFwdId.end(),
                  scopedId) != gradsProvidedForFwdId.end()) {
      auto gradId   = fwdIdToBwdGradId(scopedId);
      auto gradInfo = fwdGraph.getTensors().get(scopedId)->info;
      bwdGraph.addInput(gradId, gradInfo);
      gradTensorMap.insert({scopedId, gradId});
    }
  }

  // Add all ops in the fwdGraph to pending ops
  std::set<Op *> pendingOps;
  for (auto &id_op : fwdGraph.getOps()) {
    auto op = id_op.second.get();
    pendingOps.insert(op);
  }

  // Get all grad ops now, but don't link them up.
  for (auto &id_op : fwdGraph.getOps()) {
    auto op           = id_op.second.get();
    auto calledGraphs = op->getCalledGraphs();
    if (!op->getCalledGraphs().empty()) {
      op->setCalledSubgraphGradInfo(calledGraphsGradInfo);
    }
    gradOpStore[op] = op->getGradOps();
  }

  while (!pendingOps.empty()) {
    logging::trace("[ ] !pendingOps.empty()");
    // get all the ops that are ready to grow grad ops
    std::vector<Op *> readyOps;
    for (auto op : pendingOps) {
      if (opIsReadyToCreateGradients(op)) {
        readyOps.push_back(op);
      }
    }
    if (readyOps.empty()) {
      // we're stuck.
      std::stringstream ss;
      ss << "[Autodiff] Unable to create backwards graph for "
         << fwdGraph.getGraphString() << " "
         << "because none of the following ops are ready to grow their grad "
         << "ops:" << std::endl;

      for (auto op : pendingOps) {
        ss << " - " << op->str() << " (" << opNotReadyExplanation(op) << ")"
           << std::endl;
      }
      throw error(ss.str());
    }
    // remove ready ops from pending
    for (auto op : readyOps) {
      pendingOps.erase(pendingOps.find(op));
    }
    // grow grad ops for op
    for (auto fwdOp : readyOps) {
      auto bwdOps = growGradOps(fwdOp);

      for (auto bwdOp : bwdOps) {
        registerBwdOp(fwdOp, bwdOp);
      }
    }
  }

  // connect up outputs
  for (auto &scopedId : fwdGraph.getInputIds()) {
    if (gradTensorMap.find(scopedId) != gradTensorMap.end()) {
      auto gradId = fwdIdToBwdGradId(scopedId);
      bwdGraph.markAsOutput(gradId);
    } else {
      // We are unable to provide a gradient output for scopedId, if
      // `gradsRequiredForFwdId` is set and the scopedId is in it then the
      // user requires this gradient and being unable to provide it is an error.
      if (gradsRequiredForFwdId) {
        if (std::find(gradsRequiredForFwdId->begin(),
                      gradsRequiredForFwdId->end(),
                      scopedId) != gradsRequiredForFwdId->end()) {
          throw error("[Autodiff] Unable to provide gradient output for '{}' "
                      "in {}",
                      scopedId,
                      bwdGraph.getGraphString());
        }
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

void BackwardsGraphCreatorHelper::registerBwdOp(Op *fwdOp, Op *bwdOp) {
  for (auto &idx_tensor : bwdOp->output->tensorMap()) {
    auto bwdOutIndex = idx_tensor.first;
    auto bwdTensor   = idx_tensor.second;
    auto fwdInIndex  = bwdOp->getNonGradInIndex(bwdOutIndex);
    auto fwdTensor   = fwdOp->inTensor(fwdInIndex);
    gradRegister.insert(fwdTensor, bwdTensor);
  }

  for (auto &fwdTensor_partials : gradRegister.popComplete()) {
    auto fwdTensor = fwdTensor_partials.first;
    auto &partials = fwdTensor_partials.second;
    auto sumOp     = growGradSumOp(fwdTensor, partials);
    gradTensorMap.insert({fwdTensor->id, sumOp->outId(0)});
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

Op *BackwardsGraphCreatorHelper::growGradSumOp(
    Tensor *nonGradTensor,
    const std::vector<Tensor *> &partials) {
  auto gradId = fwdIdToBwdGradId(nonGradTensor->id);
  // TODO: T36603 Growing the grad sum with a fixed version may result
  // in suboptimal outlining (it's included as an outline attribute).
  auto gradSum = std::make_unique<SumOp>(
      Onnx::Operators::Sum_8,
      Op::Settings{bwdGraph,
                   GradGrowerSumOp::getGradSumOpNamePrefix() + "_" + gradId});
  OpId opId = bwdGraph.moveIntoGraph(std::move(gradSum));

  std::vector<TensorId> inputs;
  inputs.reserve(partials.size());
  for (auto &tensor : partials) {
    inputs.push_back(tensor->id);
  }

  std::vector<TensorId> outputs{gradId};

  bwdGraph.connectInputs(InputVecWrapper(inputs), opId);
  bwdGraph.connectOutputs(OutputVecWrapper(outputs), opId);
  Op *op = bwdGraph.getOps()[opId].get();
  op->setup();
  return op;
}

bool BackwardsGraphCreatorHelper::bwdIdIsGrad(const TensorId &id) {
  auto x = bwdGraph.removeScope(id);
  return popart::isGradId(x);
}

bool BackwardsGraphCreatorHelper::bwdIdIsNonGrad(const TensorId &id) {
  auto x = bwdGraph.removeScope(id);
  return !popart::isGradId(x);
}

TensorId BackwardsGraphCreatorHelper::fwdIdToBwdGradId(const TensorId &id) {
  auto x = fwdGraph.removeScope(id);
  x      = popart::getGradId(x);
  return bwdGraph.addScope(x);
}

TensorId BackwardsGraphCreatorHelper::bwdGradIdToFwdId(const TensorId &id) {
  auto x = bwdGraph.removeScope(id);
  x      = popart::getNonGradId(x);
  return fwdGraph.addScope(x);
}

TensorId BackwardsGraphCreatorHelper::bwdNonGradIdToFwdId(const TensorId &id) {
  auto x = bwdGraph.removeScope(id);
  return fwdGraph.addScope(x);
}

bool BackwardsGraphCreatorHelper::opIsReadyToCreateGradients(Op *op) {
  // Get our grad ops from the grad op store.
  auto gradOpStoreIt = gradOpStore.find(op);
  if (gradOpStoreIt == gradOpStore.end()) {
    throw error("Unexpectedly unable to find grad ops for {}", op->debugName());
  }

  // Check if all of the grad's inputs are available.
  for (auto &gradOp : gradOpStoreIt->second) {
    for (auto &inOutMapper : gradOp->gradInputInfo()) {
      if (!hasInputTensorId(op, inOutMapper)) {
        return false;
      }
      auto inputId = getInputTensorId(op, inOutMapper);
      if (!bwdGraph.getTensors().contains(inputId)) {
        return false;
      }
    }
  }

  return true;
}

std::string BackwardsGraphCreatorHelper::opNotReadyExplanation(Op *op) {

  // Get our grad ops from the grad op store.
  auto gradOpStoreIt = gradOpStore.find(op);
  if (gradOpStoreIt == gradOpStore.end()) {
    throw error("Unexpectedly unable to find grad ops for {}", op->debugName());
  }

  std::stringstream ss;
  bool isFirst = true;

  // Check if all of the grad's inputs are available.
  for (auto &gradOp : gradOpStoreIt->second) {
    for (auto &inOutMapper : gradOp->gradInputInfo()) {
      if (inOutMapper.type == GradOpInType::GradOut) {
        auto fwdId = op->outId(inOutMapper.iNonGrad);
        if (gradTensorMap.find(fwdId) == gradTensorMap.end()) {
          if (!isFirst) {
            ss << "and ";
          }
          ss << "it needs the gradient tensor for '" << fwdId
             << "', which has not been produced by any gradient op so far";
          isFirst = false;
        }
        continue;
      }
      auto inputId = getInputTensorId(op, inOutMapper);
      if (!bwdGraph.getTensors().contains(inputId)) {
        if (!isFirst) {
          ss << "and ";
        }
        ss << "it needs tensor '" << inputId << "' which is not available yet";
        isFirst = false;
      }
    }
  }

  return ss.str();
}

bool BackwardsGraphCreatorHelper::hasInputTensorId(
    Op *nonGradOp,
    const GradInOutMapper &inOutMapper) {
  int indexFwd      = inOutMapper.iNonGrad;
  GradOpInType type = inOutMapper.type;

  switch (type) {
  case GradOpInType::In:
  case GradOpInType::Out: {
    return true;
  }
  case GradOpInType::GradOut: {
    auto fwdId = nonGradOp->outId(indexFwd);
    auto found = gradTensorMap.find(fwdId);
    return (found != gradTensorMap.end());
  }
  default: {
    return false;
  }
  }
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
    auto bwdId = bwdGraph.addScope(fwdGraph.removeScope(fwdId));
    return bwdId;
  }

  //  (2) the OUTPUT at index 'indexFwd' of nonGradOp
  //  This will be a tensor internal to fwdGraph
  case GradOpInType::Out: {
    auto fwdId = nonGradOp->outId(indexFwd);
    auto bwdId = bwdGraph.addScope(fwdGraph.removeScope(fwdId));
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

std::vector<Op *> BackwardsGraphCreatorHelper::growGradOps(Op *nonGradOp) {
  auto nonGradOpId   = nonGradOp->id;
  auto gradOpStoreIt = gradOpStore.find(nonGradOp);
  if (gradOpStoreIt == gradOpStore.end()) {
    throw error("Unexpectedly unable to find grad ops for {}",
                nonGradOp->debugName());
  }
  auto bwdOps = std::move(gradOpStoreIt->second);
  gradOpStore.erase(gradOpStoreIt);

  std::vector<Op *> result;

  for (auto &uPtrOp : bwdOps) {
    Op *gradOp    = uPtrOp.get();
    OpId gradOpId = bwdGraph.moveIntoGraph(std::move(uPtrOp));

    // Reset priority, since fwd priority should not influence bwd priority
    //
    // TODO: Uncomment this. This prevented explicit priorities on certain
    // gradient ops being set which was necessary as a short term fix for
    // sharded training regressions seen in T17036. This could be replaced
    // once explicit priorities are no longer needed for this purpose. T17311
    // should fix this.
    //
    // gradOp->settings.schedulePriority = 0.0;

    gradOp->setScope(bwdGraph.getScope());

    if (nonGradOp->settings.recomputeType == RecomputeType::Recompute &&
        bwdGraph.getIr().autoRecomputationEnabled() &&
        bwdGraph.getIr().getSessionOptions().executionPhaseSettings.phases <
            2) {
      throw error("Grad Ops should be grown before recompute annotation");
    }

    // connect inputs of gradOp
    {
      // inputs to gradOp (to populate in this scope):
      std::map<int, std::string> m_inputs;
      for (auto &inOutMapper : gradOp->gradInputInfo()) {
        int indexGrad       = inOutMapper.iGrad;
        auto inputId        = getInputTensorId(nonGradOp, inOutMapper);
        m_inputs[indexGrad] = inputId;
      }

      bwdGraph.connectInputs(InputMapWrapper(m_inputs), gradOpId);
    }

    // connect outputs of gradOp
    {
      std::vector<TensorId> v_outputs;
      for (auto out_in : gradOp->gradOutToNonGradIn()) {
        int gradOut   = out_in.first;
        int nonGradIn = out_in.second;

        if (!nonGradOp->input->tensor(nonGradIn)) {
          throw error("Invalid configuration of gradOp {}. nonGradOp ({}) "
                      "OUTPUT {} is not defined ",
                      gradOp->debugName(),
                      nonGradOp->debugName(),
                      nonGradIn);
        }

        TensorId inId  = nonGradOp->inId(nonGradIn);
        TensorId outId = getEdgeGradId(nonGradOpId, nonGradIn);
        if (v_outputs.size() < gradOut + 1) {
          v_outputs.resize(gradOut + 1, "");
        }
        v_outputs[gradOut] = outId;
      }
      bwdGraph.connectOutputs(OutputVecWrapper(v_outputs), gradOpId);
    }
    gradOp->setup();

    result.push_back(gradOp);
  }

  return result;
}

void BackwardsGraphCreatorHelper::doPrune(Graph &graph) {
  using boost::range::find;

  const auto outputIds = graph.getOutputIds();
  auto isNotOutput     = [&outputIds](const TensorId &tId) {
    return find(outputIds, tId) == outputIds.end();
  };

  while (true) {
    // set to true if a tensor or op is removed
    bool continueLoop = false;

    auto &tensors = graph.getTensors();

    // Remove tensors that are not inputs or outputs and have no consumers
    const auto tensorIds = tensors.getAllTensorIds();
    for (auto id : tensorIds) {
      auto tensor = graph.getTensors().get(id);
      if (tensor->consumers.getTotal() == 0 && isNotOutput(id)) {
        if (tensor->hasProducer()) {
          auto producer = tensor->getProducer();
          producer->disconnectOutTensor(tensor);
        }
        tensors.remove(id);
        continueLoop = true;
      }
    }

    // Remove ops with no outputs
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

  // Remove inputs ids that have been pruned
  auto inputIds = graph.getInputIds();
  for (auto &id : inputIds) {
    if (!graph.getTensors().contains(id)) {
      auto inputIter = find(graph.getInputIds(), id);
      graph.graph_inputs.erase(inputIter);
    }
  }
}

} // namespace popart
