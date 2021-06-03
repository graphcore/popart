// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/subgraphcopyingstrategy.hpp>
#include <popart/subgraphpartitioner.hpp>
#include <popart/tensorindex.hpp>

namespace popart {
namespace liveness {

SubgraphCopyingStrategy::SubgraphCopyingStrategy() {}

SubgraphCopyingStrategy::~SubgraphCopyingStrategy() {}

void SubgraphCopyingStrategy::setIr(const Ir *_ir) { ir = _ir; }

void SubgraphCopyingStrategy::setLivenessAnalyzer(
    const LivenessAnalyzer *_liveness) {
  liveness = _liveness;
}

void SubgraphCopyingStrategy::apply() {}

OnEnterAndExitSubgraphCopyingStrategy::OnEnterAndExitSubgraphCopyingStrategy()
    : SubgraphCopyingStrategy() {}

OnEnterAndExitSubgraphCopyingStrategy::
    ~OnEnterAndExitSubgraphCopyingStrategy() {}

std::vector<size_t>
OnEnterAndExitSubgraphCopyingStrategy::getIndicesOfCopiesToInsertBeforeNode(
    const LivenessNode &node,
    const LivenessAnalyzer::PendingCopies &pendingCopies) const {
  if (!liveness)
    throw internal_error(
        "[OnEnterAndExitSubgraphCopyingStrategy] LivenessAnalyzer not set.");

  std::vector<size_t> result;

  // Before an exit node, add all associated output copies.
  if (node.getStatus() == OpStatus::Exit) {
    addMatching(node, pendingCopies, OpStatus::CopyOutput, result);
    addMatching(node, pendingCopies, OpStatus::CopyModified, result);
  }

  return result;
}

std::vector<size_t>
OnEnterAndExitSubgraphCopyingStrategy::getIndicesOfCopiesToInsertAfterNode(
    const LivenessNode &node,
    const LivenessAnalyzer::PendingCopies &pendingCopies) const {
  if (!liveness)
    throw internal_error(
        "[OnEnterAndExitSubgraphCopyingStrategy] LivenessAnalyzer not set.");

  std::vector<size_t> result;

  // After an enter node, add all associated input copies.
  if (node.getStatus() == OpStatus::Enter) {
    addMatching(node, pendingCopies, OpStatus::CopyInput, result);
  }

  return result;
}

void OnEnterAndExitSubgraphCopyingStrategy::addMatching(
    const LivenessNode &node,
    const LivenessAnalyzer::PendingCopies &pendingCopies,
    OpStatus status,
    std::vector<size_t> &result) const {
  // Add pending copies of a specific OpStatus that match the callstack of node.
  for (size_t index = 0; index < pendingCopies.size(); ++index) {
    const auto &pendingNode = pendingCopies[index];
    if (pendingNode.getStatus() == status &&
        pendingNode.getCallStack() == node.getCallStack() &&
        pendingNode.getSubgraphIndex() == node.getSubgraphIndex()) {
      result.push_back(index);
    }
  }
}

JustInTimeSubgraphCopyingStrategy::JustInTimeSubgraphCopyingStrategy()
    : SubgraphCopyingStrategy(), aliasesMap{}, isPartitionableCache() {}

JustInTimeSubgraphCopyingStrategy::~JustInTimeSubgraphCopyingStrategy() {}

void JustInTimeSubgraphCopyingStrategy::apply() {
  if (!ir)
    throw internal_error("[JustInTimeSubgraphCopyingStrategy] Ir not set.");

  for (const auto graph : ir->getAllGraphs()) {
    // Remember for each graph if it is partitionable. We cache this because
    // working it out involves iterating over all ops in the Ir, which may
    // not be quick, and we use this info a lot.
    bool partitionable = SubgraphPartitioner::isPartitionable(*graph);
    isPartitionableCache[graph->id.str()] = partitionable;
  }

  aliasesMap.setIr(ir);
  aliasesMap.update();
}

std::vector<size_t>
JustInTimeSubgraphCopyingStrategy::getIndicesOfCopiesToInsertBeforeNode(
    const LivenessNode &node,
    const LivenessAnalyzer::PendingCopies &pendingCopies) const {

  std::vector<size_t> results;

  if (node.getStatus() == OpStatus::Normal) {

    // Add CopyInputs that 'produce' tensors consumed by the op. Note that if
    // the op is in a subgraph that is unpartitionable all CopyInputs would have
    // been removed from the pendingCopies list after the associated enter node.
    results = addCopyInputsForNormalOp(node, pendingCopies);

  } else if (node.getStatus() == OpStatus::Exit) {

    // Add CopyOutput iff there are any straggling ones in the pending list
    // (as there aren't any more ops in the schedule this should flush out
    // any remaining ones. We do this 'before node' rather than after to ensure
    // CopyOutputs/CopyModifieds don't cross the Exit boundary for the CallOp.
    const auto op           = node.getOp();
    const auto calledGraphs = op->getCalledGraphs();
    const auto &graph       = *calledGraphs.at(node.getSubgraphIndex());
    const auto &sched       = liveness->getGraphOpSchedule(graph.id);

    results =
        addCopyOutputsForSchedPosition(node, pendingCopies, graph, sched.end());
  }

  return results;
}

std::vector<size_t>
JustInTimeSubgraphCopyingStrategy::getIndicesOfCopiesToInsertAfterNode(
    const LivenessNode &node,
    const LivenessAnalyzer::PendingCopies &pendingCopies) const {

  std::vector<size_t> result;

  if (!liveness)
    throw internal_error(
        "[JustInTimeSubgraphCopyingStrategy] LivenessAnalyzer not set.");

  if (node.getStatus() == OpStatus::Normal) {

    // Look at the graph of the node we're about to add to the schedule.
    // If the remaining ops in the graph have no effect on a CopyOutputs/
    // CopyModified copies then we may as well include it now. We can only only
    // do this for partitionable graphs. For unpartitionable graphs outputs need
    // to be added just before the exit node.

    const auto op     = node.getOp();
    const auto &graph = op->getGraph();

    if (isPartitionable(graph)) {

      const auto &sched = liveness->getGraphOpSchedule(graph.id);
      auto opIt         = std::find(sched.begin(), sched.end(), op);

      if (opIt == sched.end()) {
        // This shouldn't happen.
        throw internal_error("[JustInTimeSubgraphCopyingStrategy] Unable to "
                             "find op in schedule ({})",
                             (*opIt)->debugName());
      }

      result = addCopyOutputsForSchedPosition(
          node, pendingCopies, graph, std::next(opIt));
    }

  } else if (node.getStatus() == OpStatus::Enter) {

    auto op           = node.getOp();
    auto calledGraphs = op->getCalledGraphs();
    auto calledGraph  = calledGraphs.at(node.getSubgraphIndex());

    if (!isPartitionable(*calledGraph)) {
      // For unpartitionable subgraphs, we include *any* pending CopyInputs
      // here. All CopyInputs at this point are either associated with our own
      // call or with a parent call. Note that this does mean parent CopyInputs
      // could be included *after* the Enter node instead of before, but that's
      // not something that affects partitioning -- as they don't pass an op
      // the graph won't be partitioned.

      for (size_t index = 0; index < pendingCopies.size(); ++index) {
        const auto &pendingNode = pendingCopies[index];
        if (pendingNode.getStatus() == OpStatus::CopyInput) {
          result.push_back(index);
        }
      }
    }

    // NOTE: If unconsumed CopyInputs cause problems in the logic (e.g. these
    // CopyInputs are never included in the schedule and cause an exception
    // on Exit) then this is the place to look for and include such inputs. We
    // refrain from this for now because to our knowledge this shouldn't happen.

  } else if (node.getStatus() == OpStatus::Exit) {

    // Do a sanity check: ensure no copies associated with this exit node
    // are still pending -- all copies should reside between enter and exit.

    for (size_t index = 0; index < pendingCopies.size(); ++index) {
      const auto &pendingNode = pendingCopies[index];
      if (pendingNode.getCallStack() == node.getCallStack() &&
          pendingNode.getSubgraphIndex() == node.getSubgraphIndex()) {
        throw internal_error("[JustInTimeSubgraphCopyingStrategy] Copy \"{}\" "
                             "is still pending following the exit node; it was "
                             "ommitted from the schedule",
                             pendingNode);
      }
    }
  }

  return result;
}

std::vector<size_t> JustInTimeSubgraphCopyingStrategy::addCopyInputsForNormalOp(
    const LivenessNode &node,
    const LivenessAnalyzer::PendingCopies &pendingCopies) const {

  // A variable to hold the result indices.
  std::deque<size_t> indices;
  // A stack of IDs to process.
  std::deque<TensorId> idsRequiringInput;

  // For each tensor consumed by the op, add it to a stack. If there are
  // CopyInputs that 'produce' these tensors, we should insert these copies
  // before the op, as these tensor values are required.
  //
  // Moreover, for each inserted CopyInput (A), consider the CallOp associated
  // with the CopyInput. If the CallOp was the first op in its subgraph that
  // 'consumed' the tensor produced by CopyInput (A) and that tensor is itself
  // produced by a CopyInput (B), then (B) should be inserted before (A).
  //
  // We do this by looking recursively through pendingCopies. If 1) another op
  // already consumed the CopyInput or 2) the tensor is not produced by a
  // CopyInput then it will not be available in pendingCopies. The reason for 1)
  // is because this function will have already been called for said op and
  // the relevant CopyInput will have already been selected from pendingCopies
  // by the time we're in this function.

  for (auto input : node.getOp()->input->tensorIdMap()) {
    // Add to the front to consider the first input last (ensuring its copy
    // ends up first in the indices vector). This ensures that normally copies
    // are in input index-order.
    idsRequiringInput.emplace_front(input.second);
  }

  while (!idsRequiringInput.empty()) {
    // Get the front.
    auto consumedId = idsRequiringInput.front();
    // Remove front element.
    idsRequiringInput.pop_front();

    // Get CopyInputs that produce consumedId we haven't got yet.
    auto newIndices = getAssociatedCopies(consumedId,
                                          TensorPosition::Producing,
                                          pendingCopies,
                                          indices,
                                          FilterSetting::Include,
                                          FilterSetting::Ignore,
                                          FilterSetting::Ignore);

    // Error if we get more than one such input copy.
    if (!newIndices.empty()) {
      if (newIndices.size() == 1) {

        // Insert the index.
        auto index = newIndices[0];
        indices.insert(indices.begin(), index);
        const auto &copyNode = pendingCopies[index];

        // Log it.
        logging::devicex::trace("[JustInTimeSubgraphCopyingStrategy] Adding "
                                "CopyInput('{}'<-'{}') before <op> because "
                                "'{}' is required at this position ..."
                                "\n    ... with <op> being {}",
                                getProducedTensorForCopy(copyNode),
                                getConsumedTensorForCopy(copyNode),
                                consumedId,
                                node.getOp()->debugName());

        // Here we do a 'recursive' step to ensure we include the CopyInput
        // from the parent graph that 'produces' the tensor 'consumed' by the
        // CopyInput added just now, and ensure this is included *before*
        // the index just added.^1 This only applies if the tensor copied
        // by the parent CallOp to produce the input is itself a graph input in
        // the parent graph and is first consumed in said CallOp.
        //
        // [^1]: We put quotes around recursive because we've implemented this
        // with a stack and a while loop instead of actual recursion.

        auto copyConsumedId = getConsumedTensorForCopy(copyNode);
        idsRequiringInput.push_back(copyConsumedId);

      } else {
        throw error("[JustInTimeSubgraphCopyingStrategy] Unexpectedly found "
                    "multiple ({}) CopyInput copies producing for tensor '{}'",
                    newIndices.size(),
                    consumedId);
      }
    }
  }

  return std::vector<size_t>(indices.begin(), indices.end());
}

std::vector<size_t>
JustInTimeSubgraphCopyingStrategy::addCopyOutputsForSchedPosition(
    const LivenessNode &node,
    const LivenessAnalyzer::PendingCopies &pendingCopies,
    const Graph &graph,
    std::vector<Op *>::const_iterator schedPos) const {

  enum class OutputType { CopyOutput, CopyModified };

  // A variable to hold the result indices.
  std::deque<size_t> indices;
  // A stack of IDs to process, include whether it's a potential CopyOutput or
  // CopyModified.
  std::deque<std::tuple<TensorId, OutputType>> idsReadyForOutput;

  // First, determine those subgraph outputs that cannot be changed (and inputs
  // that are not modified) by any ops from schedPos onwards. An op changes
  // an output if the op either produces that tensor (not including aliases) or
  // it has some or all of the tensor as an aliased input and modifies said
  // input.
  //
  // For each tensor no longer modified at this position, add it to a stack.
  // If there are CopyOutputs that 'consume' the output tensors, insert them
  // at this position as the outputs are 'ready'.
  //
  // Moreover, for each inserted CopyOutput (A), consider the CallOp associated
  // with the CopyOutput. If the CallOp is the last op in its subgraph to
  // modify the tensor 'produced' by the CopyOutput (A) and this tensor is
  // consumed by a CopyOutput (B) itself, then we may as well include (B) in the
  // schedule after (A).

  const auto &sched = liveness->getGraphOpSchedule(graph.id);

  // Get the subgraph output tensors that are ready to be CopyOutput'ed if they
  // haven't already.

  for (const auto &outputId : graph.getOutputIds()) {
    // If it's an output produced by an op yet to come, it's not ready. Also, if
    // an alias of the tensor is a modified input of an op yet to come, it's not
    // ready.
    auto output = graph.getTensors().get(outputId);
    if (!isProducedInOpRange(schedPos, sched.end(), outputId) &&
        !isModifiedInOpRange(schedPos, sched.end(), output)) {
      idsReadyForOutput.emplace_back(outputId, OutputType::CopyOutput);
    }
  }

  for (const auto &inputId : graph.getInputIds()) {
    // If an alias of the tensor is a modified input of an op yet to come, it's
    // not ready. Inputs shouldn't be produced by an op in the subgraph.
    auto input = graph.getTensors().get(inputId);
    if (!isModifiedInOpRange(schedPos, sched.end(), input)) {
      idsReadyForOutput.emplace_back(inputId, OutputType::CopyModified);
    }
  }

  while (!idsReadyForOutput.empty()) {
    // Get the front.
    auto tensorId   = std::get<0>(idsReadyForOutput.front());
    auto outputType = std::get<1>(idsReadyForOutput.front());

    // Remove front element.
    idsReadyForOutput.pop_front();

    // Get indices of relevant pending CopyOutput/CopyModified nodes.
    std::vector<size_t> newIndices;

    if (outputType == OutputType::CopyOutput) {
      // Get CopyOutputs that 'consume' tensorIds' we haven't gotten yet.
      newIndices = getAssociatedCopies(tensorId,
                                       TensorPosition::Consuming,
                                       pendingCopies,
                                       indices,
                                       FilterSetting::Ignore,
                                       FilterSetting::Include,
                                       FilterSetting::Ignore);
    } else if (outputType == OutputType::CopyModified) {
      // Get CopyOutputs that 'consume' tensorIds' we haven't gotten yet.
      newIndices = getAssociatedCopies(tensorId,
                                       TensorPosition::Consuming,
                                       pendingCopies,
                                       indices,
                                       FilterSetting::Ignore,
                                       FilterSetting::Ignore,
                                       FilterSetting::Include);
    } else {
      throw internal_error("[JustInTimeSubgraphCopyingStrategy] Invalid value "
                           "for outputType ({})",
                           static_cast<int>(outputType));
    }

    for (auto index : newIndices) {

      // Insert the index at the back.
      indices.insert(indices.end(), index);
      const auto &copyNode = pendingCopies[index];

      // Log it.
      if (node.getStatus() == OpStatus::Normal) {
        logging::devicex::trace("[JustInTimeSubgraphCopyingStrategy] Adding "
                                "{}('{}'<-'{}') after <op> because '{}' is "
                                "is ready to copy now ..."
                                "\n    ... with <op> being {}",
                                outputType == OutputType::CopyOutput
                                    ? "CopyOutput"
                                    : "CopyModified",
                                getProducedTensorForCopy(copyNode),
                                getConsumedTensorForCopy(copyNode),
                                tensorId,
                                node.getOp()->debugName());
      } else {
        const auto op           = node.getOp();
        const auto calledGraphs = op->getCalledGraphs();
        const auto &graph       = *calledGraphs.at(node.getSubgraphIndex());
        logging::devicex::trace("[JustInTimeSubgraphCopyingStrategy] Adding "
                                "{}('{}'<-'{}') before the end of {} because "
                                "'{}' is ready to copy now",
                                outputType == OutputType::CopyOutput
                                    ? "CopyOutput"
                                    : "CopyModified",
                                getProducedTensorForCopy(copyNode),
                                getConsumedTensorForCopy(copyNode),
                                graph.getGraphString(),
                                tensorId);
      }

      // Here, akin to inputs, we do a 'recursive' step to ensure we include
      // the CopyOutput/CopyModifieds from the parent graph that are ready
      // to output as a result of the copy we just included. This only applies
      // if the tensor produced by the copy we added is:
      //
      //  1) A subgraph output in the parent graph that is last modified by
      //     the CallOp associated with the copy we just added. In this case
      //     we can add the associated CopyOutput.
      //  2) A subgraph input in the parent graph that is last modified by
      //     the CallOp associated with the copy we just added. In this case,
      //     we can add the associated CopyModified, if any.
      //
      // Unlike inputs, we cannot rely on the call order to work out whether
      // the CallOp was the *last* place a tensor was modified (for inputs, if
      // there is a node that consumes the tensor *earlier* that node's call
      // to the SugraphCopyingStrategy would have inserted the CopyInput
      // at that point already) and we have to explicitly go through the
      // remaining schedule in the CallOp's subgraph to ensure this is the
      // case, otherwise we risk copying outputs too early.

      auto producedId = getProducedTensorForCopy(copyNode);

      // Get the graph the subgraphing op is in.
      auto subgraphOp       = copyNode.getOp();
      auto &subgraphOpGraph = subgraphOp->getGraph();
      auto &subgraphOpSched = liveness->getGraphOpSchedule(subgraphOpGraph.id);

      // Don't consider propagating to CopyOuputs/CopyModifieds that belong to
      // to subgraphs that are not partitionable.
      if (isPartitionable(subgraphOpGraph)) {
        auto producedIdAliases  = getAliases(subgraphOpGraph, producedId);
        auto subgraphOpSchedPos = std::find(
            subgraphOpSched.begin(), subgraphOpSched.end(), subgraphOp);

        if (subgraphOpSchedPos == subgraphOpSched.end()) {
          // This shouldn't happen.
          throw internal_error("[JustInTimeSubgraphCopyingStrategy] Unable to "
                               "find SubgraphOp in schedule ({})",
                               subgraphOp->debugName());
        }

        // Shorthands.
        auto begin = std::next(subgraphOpSchedPos);
        auto end   = subgraphOpSched.end();

        for (auto alias : producedIdAliases) {
          auto aliasTensor = subgraphOpGraph.getTensors().get(alias);
          if (subgraphOpGraph.hasOutputId(alias)) {
            // Consider any CopyOutputs for tensors that have been produced by
            // the copy already included and are not modified or produced by
            // ops following the associated CallOp.
            if (!isProducedInOpRange(begin, end, alias) &&
                !isModifiedInOpRange(begin, end, aliasTensor)) {
              idsReadyForOutput.emplace_back(alias, OutputType::CopyOutput);
            }
          }
          if (subgraphOpGraph.hasInputId(alias)) {
            // Consider any CopyModified for tensors that have been produced by
            // the copy already included and are not modified by
            // ops following the associated CallOp.
            if (!isModifiedInOpRange(begin, end, aliasTensor)) {
              idsReadyForOutput.emplace_back(alias, OutputType::CopyModified);
            }
          }
        }
      }

      // Finally there is a specific edge case we need to accomodate. It is
      // possible (but unlikely) that the tensor 'consumed' by the
      // output was never modified or produced by an op because no such op
      // exists and it was actually produced by an input copy. In such a
      // scenario we need to schedule the input copy now.
      //
      // An example of when this happens is when a subgraph has no ops and
      // just outputs an input.

      auto consumedId   = getConsumedTensorForCopy(copyNode);
      auto inputIndices = getAssociatedCopies(consumedId,
                                              TensorPosition::Producing,
                                              pendingCopies,
                                              indices,
                                              FilterSetting::Include,
                                              FilterSetting::Ignore,
                                              FilterSetting::Ignore);

      if (inputIndices.size() == 1) {

        // Insert the index at the front, as it needs to happen first.
        auto index = inputIndices[0];
        indices.insert(indices.begin(), index);
        const auto &inputCopyNode = pendingCopies[index];

        // Log it.
        logging::devicex::trace("[JustInTimeSubgraphCopyingStrategy] "
                                "Adding CopyInput('{}'<-'{}') before a "
                                "CopyOutput or CopyModified because it "
                                "produces '{}', which is needed by an "
                                "CopyOutput or CopyModified",
                                getProducedTensorForCopy(inputCopyNode),
                                getConsumedTensorForCopy(inputCopyNode),
                                consumedId);

      } else if (inputIndices.size() >= 2) {
        // Error if we get more than one such input copy.
        throw error("[JustInTimeSubgraphCopyingStrategy] Unexpectedly found "
                    "multiple ({}) CopyInput nodes producing tensor '{}'",
                    inputIndices.size(),
                    consumedId);
      }
    }
  }

  return std::vector<size_t>(indices.begin(), indices.end());
}

std::vector<size_t> JustInTimeSubgraphCopyingStrategy::getAssociatedCopies(
    const TensorId &tensorId,
    TensorPosition tensorPos,
    const LivenessAnalyzer::PendingCopies &pendingCopies,
    const std::deque<size_t> &ignoreIndices,
    const FilterSetting copyInputFilter,
    const FilterSetting copyOutputFilter,
    const FilterSetting copyModifiedFilter) const {

  std::vector<size_t> indices;

  for (size_t rIndex = 0; rIndex < pendingCopies.size(); ++rIndex) {
    // Go in reverse order so that, when we add indices to the front, we go
    // in schedule order within one iteration.
    size_t index = pendingCopies.size() - rIndex - 1;

    const auto &pendingNode = pendingCopies[index];

    // Don't consider if already part of the result.
    auto ignoreIt =
        std::find(ignoreIndices.begin(), ignoreIndices.end(), index);

    if (ignoreIt == ignoreIndices.end()) {
      const bool isCopyInput  = pendingNode.getStatus() == OpStatus::CopyInput;
      const bool isCopyOutput = pendingNode.getStatus() == OpStatus::CopyOutput;
      const bool isCopyMod = pendingNode.getStatus() == OpStatus::CopyModified;

      if ((copyInputFilter == FilterSetting::Include && isCopyInput) ||
          (copyOutputFilter == FilterSetting::Include && isCopyOutput) ||
          (copyModifiedFilter == FilterSetting::Include && isCopyMod)) {
        auto copyTensorId = (tensorPos == TensorPosition::Producing)
                                ? getProducedTensorForCopy(pendingNode)
                                : getConsumedTensorForCopy(pendingNode);

        if (tensorId == copyTensorId) {
          // Needs to come before everything else.
          indices.push_back(index);
        }
      }
    }
  }

  return indices;
}

std::set<TensorId>
JustInTimeSubgraphCopyingStrategy::getAliases(const Graph &graph,
                                              const TensorId &id) const {

  std::set<TensorId> aliasedTensors;
  aliasedTensors.insert(id);

  Tensor *t             = graph.getTensors().get(id);
  auto &aliases         = aliasesMap.getAliases(graph.id);
  auto aliasedTensorMap = aliases.aliasChainsFrom(t);
  auto fullRegion       = view::Region::getFull(t->info.shape());
  for (auto &chain : aliasedTensorMap) {
    auto regions = chain.second.apply(fullRegion);
    if (nonEmptyRegion(regions)) {
      auto aliasedTensor = chain.first;
      aliasedTensors.insert(aliasedTensor->id);
    }
  }

  return aliasedTensors;
}

bool JustInTimeSubgraphCopyingStrategy::isProducedInOpRange(
    std::vector<Op *>::const_iterator begin,
    std::vector<Op *>::const_iterator end,
    TensorId outputId) const {
  for (auto opIt = begin; opIt != end; ++opIt) {
    for (auto i : (*opIt)->output->tensorIdMap()) {
      if (i.second == outputId)
        return true;
    }
  }
  return false;
}

bool JustInTimeSubgraphCopyingStrategy::isModifiedInOpRange(
    std::vector<Op *>::const_iterator begin,
    std::vector<Op *>::const_iterator end,
    Tensor *tensor) const {
  auto pred = [&](Tensor *alias) -> bool {
    for (auto opIt = begin; opIt != end; ++opIt) {
      for (auto i : (*opIt)->input->tensorIdMap()) {
        if ((*opIt)->modifiesIndex(i.first) && i.second == alias->id)
          return true;
      }
    }
    return false;
  };
  return tensor->anyAlias(pred);
}

TensorId JustInTimeSubgraphCopyingStrategy::getConsumedTensorForCopy(
    const LivenessNode &node) const {

  if (node.getStatus() == OpStatus::CopyInput) {
    return node.getTensorIds().first;
  } else if ((node.getStatus() == OpStatus::CopyOutput) ||
             (node.getStatus() == OpStatus::CopyModified)) {
    return node.getTensorIds().second;
  } else {
    throw error("[JustInTimeSubgraphCopyingStrategy] Node {} is not a copy",
                node);
  }
}

TensorId JustInTimeSubgraphCopyingStrategy::getProducedTensorForCopy(
    const LivenessNode &node) const {

  if (node.getStatus() == OpStatus::CopyInput) {
    return node.getTensorIds().second;
  } else if ((node.getStatus() == OpStatus::CopyOutput) ||
             (node.getStatus() == OpStatus::CopyModified)) {
    return node.getTensorIds().first;
  } else {
    throw error("[JustInTimeSubgraphCopyingStrategy] Node {} is not a copy",
                node);
  }
}

bool JustInTimeSubgraphCopyingStrategy::isPartitionable(
    const Graph &graph) const {
  return isPartitionableCache.at(graph.id.str());
}

} // namespace liveness
} // namespace popart
