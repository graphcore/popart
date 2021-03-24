// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <transforms/autodiff/backwardspasscreator.hpp>

#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/opmanager.hpp>
#include <popart/pbwrap.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>

#include <boost/range/algorithm/find.hpp>

namespace popart {

BackwardPassCreator::BackwardPassCreator(Graph &fwdGraph_, Graph &bwdGraph_)
    : fwdGraph(fwdGraph_), bwdGraph(bwdGraph_) {
  growGradGraph();

  // create map of bwdGraph input TensorIds to fwdGraph input/gradient TensorIds
  // must be done before pruning
  std::map<TensorId, TensorId> bwdInputIdToFwdTensorId;
  for (int i = 0; i < fwdGraph.getInputIds().size(); i++) {
    auto fwdIn = fwdGraph.getInputId(i);
    auto bwdIn = bwdGraph.getInputId(i);
    bwdInputIdToFwdTensorId.insert({bwdIn, fwdIn});
  }
  for (auto &fwdOut : fwdGraph.getOutputIds()) {
    auto gradId = getGradId(fwdOut);
    bwdInputIdToFwdTensorId.insert({gradId, fwdOut});
  }

  doPrune(bwdGraph);

  populateGradInInfo(bwdInputIdToFwdTensorId);
}

void BackwardPassCreator::populateGradInInfo(
    const std::map<TensorId, TensorId> &bwdInputIdToFwdTensorId) {
  // Populate bwdGraph.gradInInfo
  using boost::range::find;
  std::map<TensorId, GradInOutMapper> partialGradInfo;
  for (int i = 0; i < fwdGraph.getInputIds().size(); i++) {
    auto id = fwdGraph.getInputId(i);
    partialGradInfo.insert({id, {-1, i, GradOpInType::In}});
  }
  for (int i = 0; i < fwdGraph.getOutputIds().size(); i++) {
    auto id = fwdGraph.getOutputId(i);
    partialGradInfo.insert({id, {-1, i, GradOpInType::GradOut}});
  }

  auto bwdInputIds = bwdGraph.getInputIds();
  for (int bIdx = 0; bIdx < bwdInputIds.size(); bIdx++) {
    auto bwdId       = bwdInputIds.at(bIdx);
    auto fwdTensorId = bwdInputIdToFwdTensorId.at(bwdId);
    auto found       = partialGradInfo.find(fwdTensorId);
    if (found != partialGradInfo.end()) {
      auto gradInOutMapper  = found->second;
      gradInOutMapper.iGrad = bIdx;
      bwdGraph.gradInInfo.push_back(gradInOutMapper);
    } else {
      throw error(
          "Could not find corresponding input tensor for graph input {}",
          bwdId);
    }
  }
}

void BackwardPassCreator::growGradGraph() {
  // clone ops from the fwdGraph into the bwdGraph
  cloneGraph(fwdGraph, bwdGraph);
  // cloned outputs are not required
  for (auto &id : bwdGraph.getOutputIds()) {
    bwdGraph.removeOutput(id);
  }

  // Create an input tensor for each output tensor of fwdGraph
  for (auto &scopedId : fwdGraph.getOutputIds()) {
    auto gradId   = getGradId(scopedId);
    auto gradInfo = fwdGraph.getTensors().get(scopedId)->info;
    bwdGraph.addInput(gradId, gradInfo);
    gradTensorMap.insert({scopedId, gradId});
  }

  // Add all ops in the fwdGraph to pending ops
  std::set<Op *> pendingOps;
  for (auto &id_op : fwdGraph.getOps()) {
    auto op = id_op.second.get();
    pendingOps.insert(op);
  }

  while (!pendingOps.empty()) {
    // get all the ops that are ready to grow grad ops
    std::vector<Op *> readyOps;
    for (auto op : pendingOps) {
      if (opIsReadyToCreateGradients(op)) {
        readyOps.push_back(op);
      }
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
    if (gradTensorMap.find(scopedId) == gradTensorMap.end()) {
      throw error("Could not find tensor {} in gradTensorMap", scopedId);
    }
    auto gradId = getGradId(scopedId);
    bwdGraph.markAsOutput(gradId);
  }
}

void BackwardPassCreator::cloneGraph(const Graph &from, Graph &to) {
  to.copyFrom(from);
}

void BackwardPassCreator::registerBwdOp(Op *fwdOp, Op *bwdOp) {
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

Op *BackwardPassCreator::growGradSumOp(Tensor *nonGradTensor,
                                       const std::vector<Tensor *> &partials) {
  std::unique_ptr<popart::Op> gradSum = OpManager::createOp(
      Domain::ai_onnx,
      "Sum",
      bwdGraph.getIr().getOpSetVersionFromModel(Domain::ai_onnx),
      bwdGraph,
      "GradSum");

  OpId opId = bwdGraph.moveIntoGraph(std::move(gradSum));

  std::vector<TensorId> inputs;
  inputs.reserve(partials.size());
  for (auto &tensor : partials) {
    inputs.push_back(tensor->id);
  }

  auto gradId = getGradId(nonGradTensor->id);

  std::vector<TensorId> outputs{gradId};

  bwdGraph.connectInputs(InputVecWrapper(inputs), opId);
  bwdGraph.connectOutputs(OutputVecWrapper(outputs), opId);
  Op *op = bwdGraph.getOps()[opId].get();
  op->setup();
  return op;
}

TensorId BackwardPassCreator::getGradId(const TensorId &id) {
  auto x = fwdGraph.removeScope(id);
  x      = popart::getGradId(x);
  return bwdGraph.addScope(x);
}

bool BackwardPassCreator::opIsReadyToCreateGradients(Op *op) {
  for (auto output : op->output->tensors()) {
    if (gradTensorMap.find(output->id) == gradTensorMap.end()) {
      return false;
    }
  }

  return true;
}

std::vector<Op *> BackwardPassCreator::growGradOps(Op *nonGradOp) {
  auto nonGradOpId = nonGradOp->id;
  auto bwdOps      = nonGradOp->getGradOps();
  if (bwdOps.empty()) {
    throw error("Cannot get gradients for {}", nonGradOp->debugName());
  }

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

        int indexGrad     = inOutMapper.iGrad;
        int indexFwd      = inOutMapper.iNonGrad;
        GradOpInType type = inOutMapper.type;

        // the input at index 'indexGrad' to gradOp is
        switch (type) {
        //  (1) the INPUT at index 'indexFwd' of nonGradOp
        //  This will be a tensor internal to fwdGraph
        case GradOpInType::In: {
          auto fwdId          = nonGradOp->inId(indexFwd);
          auto bwdId          = bwdGraph.addScope(fwdGraph.removeScope(fwdId));
          m_inputs[indexGrad] = bwdId;
          break;
        }

        //  (2) the OUTPUT at index 'indexFwd' of nonGradOp
        //  This will be a tensor internal to fwdGraph
        case GradOpInType::Out: {
          auto fwdId          = nonGradOp->outId(indexFwd);
          auto bwdId          = bwdGraph.addScope(fwdGraph.removeScope(fwdId));
          m_inputs[indexGrad] = bwdId;
          break;
        }

        //  (3) the GRADIENT of the OUTPUT
        //      at index 'indexFwd' of nonGradOp.
        case GradOpInType::GradOut: {
          auto fwdId = nonGradOp->outId(indexFwd);
          auto found = gradTensorMap.find(fwdId);
          if (found == gradTensorMap.end()) {
            throw error("Could not find TensorId '{}' in gradTensorMap", fwdId);
          }
          m_inputs[indexGrad] = found->second;
          break;
        }
        }
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
        TensorId outId = getEdgeGradId(inId, nonGradOpId, nonGradIn);
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

void BackwardPassCreator::doPrune(Graph &graph) {
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
