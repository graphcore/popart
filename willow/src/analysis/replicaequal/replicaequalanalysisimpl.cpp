// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include <analysis/replicaequal/replicaequalanalysisimpl.hpp>

#include <popart/alias/aliasmodelgrower.hpp>
#include <popart/analysis/replicaequal/replicaequalanalysis.hpp>
#include <popart/graph.hpp>
#include <popart/logging.hpp>
#include <popart/op/getrandomseed.hpp>
#include <popart/pointercomparators.hpp>

namespace popart {

ReplicaEqualAnalysisImpl::ReplicaEqualAnalysisImpl(const Ir &ir_)
    : ReplicaEqualAnalysisProxy{}, localAliasModel{}, ir{ir_},
      aliasModel{localAliasModel}, analysisResults{}, graphSchedules{} {}

ReplicaEqualAnalysisImpl::ReplicaEqualAnalysisImpl(const Ir &ir_,
                                                   AliasModel &aliasModel_)
    : ReplicaEqualAnalysisProxy{}, localAliasModel{}, ir{ir_},
      aliasModel{aliasModel_}, analysisResults{}, graphSchedules{} {}

void ReplicaEqualAnalysisImpl::apply() {

  // TODO(T48752): Remove temporary switch that default disables this.
  if (ir.get().getSessionOptions()._enableRngStateManagement) {

    initialise();

    // Initialise values for variables, streams and consts.
    addVariableTensorsToAnalysisResults();
    addStreamTensorsToAnalysisResults();

    for (auto graph : ir.get().getAllGraphs()) {
      addConstTensorsToAnalysisResults(graph);
    }

    auto doAnalysisIter = [&]() {
      // Clear flag so we'll detect changes from this point.
      analysisResults.clearChanges();
      // Do one forward propagation.
      fwdPropagateIsReplicaEqualThroughGraph(&ir.get().getMainGraph(), {});
      // Deal with tensors aliased to variable tensors that have changed.
      processMainGraphAliases();
    };

    // Loop until fixpoint.
    while (true) {

      // Do an analysis iteration.
      doAnalysisIter();

      if (!analysisResults.hasChanged()) {
        // We're done.
        break;
      }
    }

    // Do one more iteration. The next iteration shouldn't change anything
    // result-wise, but we use it to show up any disagreements between call
    // sites that we should warn our users about. This happens if in our result
    // set a tensor is assigned `false` (not replica-equal) but in some code
    // path we observe it is equal and call `setValueAt(..., true)` for that
    // tensor. It will resolve to false and hence be unchanged, but there is
    // disagreement.
    doAnalysisIter();

    if (analysisResults.hasDisagreements()) {
      for (auto tensor : analysisResults.getDisagreements()) {
        logging::warn(
            "[ReplicaEqualAnalysis] There is disagreement between code "
            "paths involving tensor '{}' as to whether its value is "
            "identical between replicas or not. Because the same code "
            "is lowered for all code paths we can't guarantee its values "
            "will be identical between replicas for any of these code "
            "paths. This problem could originate from the tensors being "
            "in a subgraph that is instantiated by several CallOps, "
            "IfOps or LoopOps that are not compatible with each other "
            "in this sense.",
            tensor->id);
      }
    }
  }
}

IsReplicaEqual ReplicaEqualAnalysisImpl::isOpInputEqual(const Op *op,
                                                        InIndex inIndex) const {
  if (!op->hasInput(inIndex)) {
    throw internal_error("[ReplicaEqualAnalysis] '{}' does not have an input "
                         "at index {}",
                         op->debugName(),
                         inIndex);
  }

  // Get the analysis result for a specific Op input.
  auto tensor = op->inTensor(inIndex);
  return analysisResults.getValueBefore(tensor, op);
}

IsReplicaEqual
ReplicaEqualAnalysisImpl::isOpOutputEqual(const Op *op,
                                          OutIndex outIndex) const {
  if (!op->hasOutput(outIndex)) {
    throw internal_error("[ReplicaEqualAnalysis] '{}' does not have an output "
                         "at index {}",
                         op->debugName(),
                         outIndex);
  }

  // Get the analysis result for a specific Op output.
  auto tensor = op->outTensor(outIndex);
  return analysisResults.getValueAt(tensor, op);
}

std::map<std::string, popart::any>
ReplicaEqualAnalysisImpl::getOpAttrs(const Op *op) const {
  std::map<std::string, popart::any> attrs;
  // TODO(T48752): Remove temporary switch that default disables this.
  if (ir.get().getSessionOptions()._enableRngStateManagement) {
    for (const auto &in : op->input->tensorMap()) {
      attrs["replEqIn" + std::to_string(in.first)] =
          isOpInputEqual(op, in.first);
    }
    for (const auto &out : op->output->tensorMap()) {
      attrs["replEqOut" + std::to_string(out.first)] =
          isOpOutputEqual(op, out.first);
    }
  }
  return attrs;
}

ReplEqModifiedInputMap ReplicaEqualAnalysisImpl::getModifiedInputMapFromAliases(
    const Op *op,
    const ReplEqOutputMap &replEqOpOutputMap) const {

  // Find modified inputs and determine if they are 1) now possibly not replica
  // equal 2) now definitely replica-equal. This generic implementation looks
  // for outputs (for which we have replica-equal values) that alias the inputs
  // that are modified. We say a modified input may be not replica-equal iff
  // an output is deemed not replica-equal and that output (partially) aliases
  // the input. We say a modified input must be replica-equal iff an output
  // is deemed replica-equal and the modified input is wholly aliased in the
  // output. Note that this implementation is not 100% accurate for all ops
  // meaning that some Ops may need a specialised implementation.
  ReplEqModifiedInputMap modifiedInputs;
  for (const auto &input : op->input->tensorMap()) {
    // Ignore this index if it is not modified.
    if (op->modifiesIndex(input.first)) {

      bool haveReplEqAlias = false;

      // Look for an output that wholly aliases the input.
      for (const auto &output : op->output->tensorMap()) {

        // Ignore outputs that are not replica-equal as we can't work out
        // the input is replica-equal from a non-replica equal output alias.
        if (replEqOpOutputMap.at(output.first)) {

          // Look for an output that wholly aliases the input (e.g. there is
          // no input element that is not aliased by the output).
          const bool contains =
              aliasModel.get().contains(*output.second, *input.second);

          if (contains) {
            // Add the input to the list and work on the next input.
            modifiedInputs[input.first] = true;
            haveReplEqAlias             = true;
            continue;
          }
        }
      }

      if (!haveReplEqAlias) {
        modifiedInputs[input.first] = false;
      }
    }
  }

  return modifiedInputs;
}

std::tuple<ReplEqOutputMap, ReplEqModifiedInputMap>
ReplicaEqualAnalysisImpl::fwdPropagateIsReplicaEqualThroughGraph(
    const Graph *graph,
    const ReplEqInputMap &replEqGraphInputMap) {

  // Add graph inputs, determine if they are the same as we've already set,
  // and, if so, remember this so we can avoid the forward propagation.
  bool inputsChanged =
      addGraphInputValsToAnalysisResults(graph, replEqGraphInputMap);

  // No need to propagate ops for subgraphs we already got results for if inputs
  // are unchanged. This only holds based on the the assumption it doesn't read
  // change result based on variable tensor changes, as changes in variables
  // aren't reflected in `inputsChanged. This only holds for subgraphs (hence
  // the exclusion of main graph). This is akin to caching results for
  // subgraphs. We do this as an optimisation. Normally it shouldn't make too
  // much difference but the cost of not optimizing this could be exponential
  // in the number of call ops for some IRs.
  if ((graph->id == ir.get().getMainGraph().id) || inputsChanged) {
    auto &schedule = graphSchedules.at(graph->id);

    // Go in schedule order, otherwise results for Op inputs may not exist.
    for (auto op : schedule) {

      // Get replica equal values for Op inputs.
      auto opInputMap = getOpInputValsFromAnalysisResults(op);

      // Get replica equal values for Op outputs.
      auto res =
          op->fwdPropagateIsReplicaEqual(aliasModel.get(), opInputMap, *this);

      Updates opUpdates;
      addOpOutputValsToAnalysisResults(op, std::get<0>(res), opUpdates);
      addOpModifiedInputValsToAnalysisResults(op, std::get<1>(res), opUpdates);
      addAliasesValsToAnalysisResults(op, opUpdates);
    }
  }

  // Get replica-equalness of graph outputs.
  auto outputVals = getGraphOutputValsFromAnalysisResults(graph);
  // Get replica-equalness of graph inputs. We do this because this graph might
  // be in, e.g., a CallOp, which allows modification of inputs, in which case
  // the CallOp needs to know the final replica-equalness of inputs.
  auto inputVals = getGraphInputValsFromAnalysisResults(graph);

  return {outputVals, inputVals};
}

void ReplicaEqualAnalysisImpl::initialise() {

  // Determine and store graph schedules and alias model.
  for (auto graph : ir.get().getAllGraphs()) {
    logging::trace("[ReplicaEqualAnalysis] Initialising graph {}",
                   graph->getGraphString());

    // Get some graph schedule for all graphs and put it in graphSchedules.
    graphSchedules[graph->id] =
        graph->getOpSchedule({}, RequireOptimalSchedule::No);

    // Make sure alias models have been grown all graphs (by growing the alias
    // model for those graphs that don't currently have all tensors mapped).
    auto graphTensorIds = graph->getTensors().getAllTensorIds();
    if (std::any_of(graphTensorIds.begin(),
                    graphTensorIds.end(),
                    [&](TensorId &id) -> bool {
                      return !aliasModel.get().contains(id);
                    })) {
      logging::trace("[ReplicaEqualAnalysis] Growing alias model for graph {}",
                     graph->getGraphString());
      AliasModelGrower aliasModelGrower{aliasModel.get()};
      aliasModelGrower.growFullGraph(*graph, DataDependenciesOnly::No);
    }
  }

  // Pass graph schedules to analysis results.
  analysisResults.setGraphSchedules(graphSchedules);
}

void ReplicaEqualAnalysisImpl::addVariableTensorsToAnalysisResults() {

  for (auto tensor :
       ir.get().getMainGraph().getTensors().getOfType(TensorType::Variable)) {
    // Set for initialisation time.
    if (analysisResults.setValueAt(tensor, nonstd::nullopt, true)) {
      logging::trace(
          "[ReplicaEqualAnalysis] Setting '{}' ({}) to 1 (time: init)",
          tensor->id,
          tensor->tensorType());
    }
  }
}

void ReplicaEqualAnalysisImpl::addStreamTensorsToAnalysisResults() {

  for (auto tensor :
       ir.get().getMainGraph().getTensors().getOfType(TensorType::Stream)) {
    // Ignore tensors that are produced by host_loads.
    if (!tensor->hasProducer()) {
      // Is replica equal if mode is broadcast.
      bool initValue = (tensor->getReplicatedStreamMode() ==
                        ReplicatedStreamMode::Broadcast);
      if (analysisResults.setValueAt(tensor, nonstd::nullopt, initValue)) {
        logging::trace(
            "[ReplicaEqualAnalysis] Setting '{}' ({}) to {} (time: init)",
            tensor->id,
            tensor->tensorType(),
            initValue);
      }
    }
  }
}

void ReplicaEqualAnalysisImpl::addConstTensorsToAnalysisResults(
    const Graph *graph) {

  for (auto tensor : graph->getTensors().getOfType(TensorType::Const)) {
    // Is replica equal unless it's the streamed seed tensor.
    bool initValue = (tensor->id != GetRandomSeedOp::getStreamedSeedTensorId());
    if (analysisResults.setValueAt(tensor, nonstd::nullopt, initValue)) {
      logging::trace(
          "[ReplicaEqualAnalysis] Setting '{}' ({}) to {} (time: init)",
          tensor->id,
          tensor->tensorType(),
          initValue);
    }
  }
}

void ReplicaEqualAnalysisImpl::addOpOutputValsToAnalysisResults(
    const Op *op,
    const ReplEqOutputMap &isReplEqMap,
    Updates &updates) {
  for (const auto &entry : op->output->tensorMap()) {
    auto index        = entry.first;
    auto tensor       = entry.second;
    auto findOutputIt = isReplEqMap.find(index);
    if (findOutputIt == isReplEqMap.end()) {
      throw internal_error("[ReplicaEqualAnalysisImpl] Expected value for "
                           "output {} ('{}') in ReplEqInputMap for Op {}",
                           index,
                           op->outId(index),
                           op->debugName());
    }

    // Update actual output tensor.
    auto replEq = findOutputIt->second;
    updates.push_back({tensor, replEq});
    if (analysisResults.setValueAt(tensor, op, replEq)) {
      logging::trace("[ReplicaEqualAnalysis] Setting '{}' ({}) to {} (time: "
                     "[{}], output)",
                     tensor->id,
                     tensor->tensorType(),
                     replEq,
                     op->str());
    }
  }
}

void ReplicaEqualAnalysisImpl::addOpModifiedInputValsToAnalysisResults(
    const Op *op,
    const ReplEqModifiedInputMap &isReplEqMap,
    Updates &updates) {
  for (const auto &entry : op->input->tensorMap()) {
    if (op->modifiesIndex(entry.first)) {
      auto index       = entry.first;
      auto tensor      = entry.second;
      auto findInputIt = isReplEqMap.find(index);
      if (findInputIt == isReplEqMap.end()) {
        throw internal_error("[ReplicaEqualAnalysisImpl] Expected value for "
                             "modified input {} ('{}') in "
                             "ReplEqModifiedInputMap for Op {}",
                             index,
                             op->inId(index),
                             op->debugName());
      }

      auto replEq = findInputIt->second;
      updates.push_back({tensor, replEq});
      if (analysisResults.setValueAt(tensor, op, replEq)) {
        logging::trace("[ReplicaEqualAnalysis] Setting '{}' ({}) to {} ("
                       "time: [{}], modified input)",
                       tensor->id,
                       tensor->tensorType(),
                       replEq,
                       op->str());
      }
    }
  }
}

bool ReplicaEqualAnalysisImpl::addGraphInputValsToAnalysisResults(
    const Graph *graph,
    const ReplEqInputMap &inputMap) {

  bool changed = false;

  // Check the inputMap is the right size.
  if (inputMap.size() != graph->getInputIds().size()) {
    throw internal_error("[ReplicaEqualAnalysisImpl] Expected {} input(s) in "
                         "ReplEqInputMap for '{}' (got {})",
                         graph->getInputIds().size(),
                         graph->getGraphString(),
                         inputMap.size());
  }

  // Populate each input arg.
  for (InIndex i = 0; i < graph->getInputIds().size(); ++i) {
    auto inId        = graph->getInputId(i);
    auto findInputIt = inputMap.find(i);
    if (findInputIt == inputMap.end()) {
      throw internal_error("[ReplicaEqualAnalysisImpl] Expected value for "
                           "input {} ('{}') in ReplEqInputMap",
                           i,
                           inId);
    }

    // Set it.
    auto newValue = findInputIt->second;
    auto tensor   = graph->getInputTensor(i);

    if (analysisResults.setValueAt(tensor, nonstd::nullopt, newValue)) {
      // Log it.
      logging::trace("[ReplicaEqualAnalysis] Setting '{}' ({}) to {} (time: "
                     "init)",
                     graph->getInputTensor(i)->id,
                     graph->getInputTensor(i)->tensorType(),
                     newValue);

      changed = true;
    }
  }

  return changed;
}

void ReplicaEqualAnalysisImpl::addAliasesValsToAnalysisResults(
    const Op *op,
    const Updates &updates) {

  // Set used to ensure we do not update tensors twice.
  std::set<Tensor *, PTensorCmp> processedAliases;

  // Mark any directly updated tensors as already 'processed'. This should stop
  // processing aliases to self.
  for (const auto &update : updates) {
    auto updatedTensor = std::get<0>(update);
    processedAliases.insert(updatedTensor);
  }

  // For any aliased tensors that are wholly contained by a tensor that is
  // updated to something replica-equal, mark them as replica-equal.
  for (const auto &update : updates) {
    auto updatedTensor    = std::get<0>(update);
    auto updatedReplEqVal = std::get<1>(update);

    if (updatedReplEqVal && aliasModel.get().contains(updatedTensor->id)) {
      for (auto alias : aliasModel.get().allAliases(*updatedTensor)) {
        if (processedAliases.find(alias) == processedAliases.end()) {
          if (aliasModel.get().contains(*updatedTensor, *alias)) {
            if (analysisResults.setValueAt(alias, op, true)) {
              logging::trace("[ReplicaEqualAnalysis] Setting '{}' ({}) to {} "
                             "because all allocation elements of '{}' are also "
                             "in '{}', which was updated to {} (time: <{}>)",
                             alias->id,
                             alias->tensorType(),
                             true,
                             alias->id,
                             updatedTensor->id,
                             true,
                             op->str());
            }
          }
          // Remember we've processed this alias.
          processedAliases.insert(alias);
        }
      }
    }
  }

  // For any aliases to tensors that are thought to be not replica-equal, assume
  // alias is also not replica-equal.
  for (const auto &update : updates) {
    auto updatedTensor    = std::get<0>(update);
    auto updatedReplEqVal = std::get<1>(update);

    if (!updatedReplEqVal && aliasModel.get().contains(updatedTensor->id)) {
      for (auto alias : aliasModel.get().allAliases(*updatedTensor)) {
        if (processedAliases.find(alias) == processedAliases.end()) {
          if (analysisResults.setValueAt(alias, op, false)) {
            logging::trace("[ReplicaEqualAnalysis] Setting '{}' ({}) to {} "
                           "because it aliases '{}', which was updated to {} "
                           "(time: <{}>)",
                           alias->id,
                           alias->tensorType(),
                           false,
                           updatedTensor->id,
                           false,
                           op->str());
          }
          // Remember we've processed this alias.
          processedAliases.insert(alias);
        }
      }
    }
  }

  // Remaining aliases are those tensors aliased only to tensors that are
  // updated to replica-equal but are not wholly contained. It's probably
  // safe to assume that if they are replica-equal before, they will be
  // replica-equal still, and if they are not replica-equal, we can't guarantee
  // they are replica-equal after, so we'd err on the side of them remaining
  // not replica-equal. So let's do nothing, but let's log them.

  for (const auto &update : updates) {
    auto updatedTensor    = std::get<0>(update);
    auto updatedReplEqVal = std::get<1>(update);
    for (auto alias : aliasModel.get().allAliases(*updatedTensor)) {
      if (processedAliases.find(alias) == processedAliases.end()) {
        logging::trace("[ReplicaEqualAnalysis] Although '{}' ({}) was updated "
                       "to {} and '{}' aliases '{}' we are unable to determine "
                       "if '{}' is replica-equal following this update. We "
                       "therefore leave the value of '{}' unchanged",
                       updatedTensor->id,
                       updatedTensor->tensorType(),
                       updatedReplEqVal,
                       updatedTensor->id,
                       alias->id,
                       alias->id,
                       alias->id);
      }
    }
  }
}

ReplEqInputMap ReplicaEqualAnalysisImpl::getOpInputValsFromAnalysisResults(
    const Op *op) const {

  ReplEqInputMap result;
  for (auto entry : op->input->tensorMap()) {
    int index      = entry.first;
    Tensor *tensor = entry.second;
    result[index]  = analysisResults.getValueBefore(tensor, op);
  }
  return result;
}

ReplEqOutputMap ReplicaEqualAnalysisImpl::getGraphOutputValsFromAnalysisResults(
    const Graph *graph) const {

  ReplEqOutputMap result;
  for (OutIndex o = 0; o < graph->getOutputIds().size(); ++o) {
    result[o] = analysisResults.getFinalValue(graph->getOutputTensor(o));
  }
  return result;
}

ReplEqModifiedInputMap
ReplicaEqualAnalysisImpl::getGraphInputValsFromAnalysisResults(
    const Graph *graph) const {

  ReplEqModifiedInputMap result;
  for (InIndex i = 0; i < graph->getInputIds().size(); ++i) {
    result[i] = analysisResults.getFinalValue(graph->getInputTensor(i));
  }
  return result;
}

void ReplicaEqualAnalysisImpl::processMainGraphAliases() {
  for (auto var : ir.get().getMainGraph().getTensors().getAll()) {

    if (var->tensorType() == TensorType::Variable) {
      // Get the variable's init value.
      IsReplicaEqual varInitValue =
          analysisResults.getValueAt(var, nonstd::nullopt);

      // Get the variable's final value.
      IsReplicaEqual varFinalValue = analysisResults.getFinalValue(var);

      // If the final value is false, then the init value should be false also.
      if (varInitValue && !varFinalValue) {
        if (analysisResults.setValueAt(var, nonstd::nullopt, false)) {
          logging::trace("[ReplicaEqualAnalysis] Setting the initial value "
                         "of '{}' ({}) to {} it's final value was {} (time: "
                         "init)",
                         var->id,
                         var->tensorType(),
                         false,
                         false);
        }
      }
    }
  }
}

} // namespace popart
