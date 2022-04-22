// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <map>
#include <memory>
#include <ostream>
#include <set>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/liveness.hpp>
#include <popart/op/loop.hpp>
#include <popart/scheduler_requireoptimal.hpp>
#include <popart/subgraphcopyingstrategy.hpp>

#include "popart/error.hpp"
#include "popart/graphid.hpp"
#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/region.hpp"
#include "popart/tensor.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/tensorindex.hpp"
#include "popart/tensors.hpp"
#include "popart/vendored/optional.hpp"

namespace popart {
namespace liveness {

ExecutionContext sanitizeExecutionContext(ExecutionContext context) {
  return context == ExecutionContext::Subgraph ? ExecutionContext::Normal
                                               : context;
}

LivenessNode::LivenessNode(const OpStatus status_,
                           int index_,
                           SubgraphIndex subgraphIndex_,
                           bool isDuplicate_)
    : callStack({}), status(status_), index(index_),
      subgraphIndex(subgraphIndex_), isDuplicate(isDuplicate_) {}

LivenessNode::LivenessNode(const CallStack &callStack_,
                           const OpStatus status_,
                           int index_,
                           SubgraphIndex subgraphIndex_,
                           bool isDuplicate_)
    : callStack(callStack_), status(status_), index(index_),
      subgraphIndex(subgraphIndex_), isDuplicate(isDuplicate_) {
  setTensorIds();
  setUsedTensorIds();
}

void LivenessNode::setUsedTensorIds() {
  auto op = getOp();

  switch (status) {
  case OpStatus::CopyInput:
  case OpStatus::CopyLoopCarried:
  case OpStatus::CopyModified:
  case OpStatus::CopyOutput: {
    usedIds.insert(getTensorIds().first);
    usedIds.insert(getTensorIds().second);
    break;
  }
  case OpStatus::Normal: {
    for (auto &input : getOp()->input->tensorIdMap()) {
      usedIds.insert(input.second);
    }
    for (auto &output : getOp()->output->tensorIdMap()) {
      usedIds.insert(output.second);
    }
    break;
  }
  case OpStatus::Enter: {
    auto calledGraphs = op->getCalledGraphs();
    auto calledGraph  = calledGraphs.at(getSubgraphIndex());

    for (auto &input : op->input->tensorIdMap()) {
      auto sgInIndex =
          op->opInToSubgraphInIndex(getSubgraphIndex(), input.first);

      if (sgInIndex > -1 && sgInIndex < calledGraph->getInputIds().size() &&
          calledGraph->getTensors()
              .get(calledGraph->getInputId(sgInIndex))
              ->isLoopTripCounter()) {
        // Loop trip count is not copied
        usedIds.insert(input.second);
      }

      if (sgInIndex < 0 || sgInIndex >= calledGraph->getInputIds().size()) {
        // Op input which is not a subgraph input
        usedIds.insert(input.second);
      }
    }

    auto inputIds = calledGraph->getInputIds();
    for (int64_t i = 0; i < inputIds.size(); ++i) {
      auto opInIndex = op->subgraphInToOpInIndex(getSubgraphIndex(), i);

      if (calledGraph->getTensors().get(inputIds.at(i))->isLoopTripCounter()) {
        // Loop trip count is not copied
        usedIds.insert(inputIds.at(i));
      }

      if (!op->hasInput(opInIndex)) {
        // Subgraph input which is not an op input
        usedIds.insert(inputIds.at(i));
      }
    }
    break;
  }
  case OpStatus::Exit: {
    auto calledGraphs = op->getCalledGraphs();
    auto calledGraph  = calledGraphs.at(getSubgraphIndex());

    for (auto &output : op->output->tensorIdMap()) {
      auto sgOutIndex =
          op->opOutToSubgraphOutIndex(getSubgraphIndex(), output.first);
      if (sgOutIndex < 0 || sgOutIndex >= calledGraph->getOutputIds().size()) {
        // Op output which is not a subgraph output
        usedIds.insert(output.second);
      }
    }

    auto outputIds = calledGraph->getOutputIds();
    for (int64_t i = 0; i < outputIds.size(); ++i) {
      auto opOutIndex = op->subgraphOutToOpOutIndex(getSubgraphIndex(), i);
      if (!op->output->hasIndex(opOutIndex)) {
        // Subgraph output which is not an op output
        usedIds.insert(outputIds.at(i));
      }
    }
    break;
  }
  default:
    break;
  }
}

void LivenessNode::setTensorIds() {
  auto op = getOp();

  switch (status) {
  case OpStatus::CopyInput:
  case OpStatus::CopyModified: {
    auto calledGraphs = op->getCalledGraphs();
    auto calledGraph  = calledGraphs.at(getSubgraphIndex());

    auto sgInIdx    = op->opInToSubgraphInIndex(getSubgraphIndex(), index);
    TensorId sgInId = calledGraph->getInputId(sgInIdx);
    Tensor *sgInT   = calledGraph->getTensors().get(sgInId);

    TensorId sgTensorId = sgInId;
    if (sgInT->isExplicitLoopInput()) {
      // Explicit loop inputs (except trip count)
      // are copied to the body output tensors rather than the
      // body input tensors
      auto sgOutIdx = sgInIdx - 1;
      if (sgOutIdx >= 0 && sgOutIdx < calledGraph->getOutputIds().size()) {
        TensorId sgOutId = calledGraph->getOutputId(sgOutIdx);
        sgTensorId       = sgOutId;
      }
    }

    if (sgInIdx > -1 && sgInIdx < calledGraph->getInputIds().size()) {
      tensorIds = std::make_pair(getOp()->inId(index), sgTensorId);
    } else {
      tensorIds = std::make_pair("", "");
    }
  } break;
  case OpStatus::CopyOutput: {
    auto calledGraphs = op->getCalledGraphs();
    auto calledGraph  = calledGraphs.at(getSubgraphIndex());

    auto sgOutIdx = op->opOutToSubgraphOutIndex(getSubgraphIndex(), index);
    if (sgOutIdx > -1 && sgOutIdx < calledGraph->getOutputIds().size()) {
      tensorIds = std::make_pair(getOp()->outId(index),
                                 calledGraph->getOutputId(sgOutIdx));
    } else {
      tensorIds = std::make_pair("", "");
    }
  } break;
  case OpStatus::CopyLoopCarried: {
    auto calledGraphs = op->getCalledGraphs();
    auto calledGraph  = calledGraphs.at(getSubgraphIndex());

    // Loop carried dependencies
    auto sgOutIdx = index;
    auto sgInIdx  = index + 1;
    tensorIds     = std::make_pair(calledGraph->getOutputId(sgOutIdx),
                               calledGraph->getInputId(sgInIdx));
  } break;
  case OpStatus::Normal:
  case OpStatus::Enter:
  case OpStatus::Exit:
  default:
    tensorIds = std::make_pair("", "");
  }
}

bool LivenessNode::isProducerOf(Tensor *t) const {
  auto op = getOp();

  switch (status) {
  case OpStatus::CopyInput: {
    // Produces tensor inside subgraph
    return t->id == getTensorIds().second;
  }
  case OpStatus::CopyLoopCarried: {
    // First: output of iteration #0
    // Second: input to iteration #1 <- produced
    return t->id == getTensorIds().second;
  }
  case OpStatus::CopyModified:
  case OpStatus::CopyOutput: {
    // Produces tensor outside subgraph
    return t->id == getTensorIds().first;
  }
  case OpStatus::Normal: {
    return t->hasProducer() && t->getProducer() == getOp();
  }
  case OpStatus::Enter: {
    auto calledGraphs = op->getCalledGraphs();
    auto calledGraph  = calledGraphs.at(getSubgraphIndex());

    // Produces tensor inside subgraph (if no CopyInput exists for this input)
    auto inputIds = calledGraph->getInputIds();
    auto it       = std::find(inputIds.begin(), inputIds.end(), t->id);

    // Check if the tensor is an input tensor of the called graph
    if (it != inputIds.end()) {
      InIndex sgInIndex = std::distance(inputIds.begin(), it);

      if (calledGraph->getTensors().get(t->id)->isLoopTripCounter()) {
        // Loop trip count is not copied
        return true;
      }

      InIndex opInIndex =
          op->subgraphInToOpInIndex(getSubgraphIndex(), sgInIndex);
      if (!op->hasInput(opInIndex)) {
        // Is a subgraph input, but not an op input
        return true;
      }
    }
    return false;
  }
  case OpStatus::Exit: {
    auto calledGraphs = op->getCalledGraphs();
    auto calledGraph  = calledGraphs.at(getSubgraphIndex());

    // Produces tensor outside subgraph (no CopyOutput)
    if (t->hasProducer() && t->getProducer() == getOp()) {
      OutIndex opOutIndex = getOp()->output->indices(t).front();
      OutIndex sgOutIndex =
          op->opOutToSubgraphOutIndex(getSubgraphIndex(), opOutIndex);
      if (sgOutIndex < 0 || sgOutIndex >= calledGraph->getOutputIds().size()) {
        // Is an op output, but not a subgraph output
        return true;
      }
    }
    return false;
  }
  default:
    return false;
  }
}

bool LivenessNode::isConsumerOf(Tensor *t) const {
  auto op = getOp();

  switch (status) {
  case OpStatus::CopyInput: {
    // Consumes tensor outside subgraph
    return t->id == getTensorIds().first;
  }
  case OpStatus::CopyLoopCarried: {
    // First: output of iteration #0 <- consumed
    // Second: input to iteration #1
    return t->id == getTensorIds().first;
  }
  case OpStatus::CopyModified: {
    // Consumes tensor inside subgraph
    return t->id == getTensorIds().second;
  }
  case OpStatus::CopyOutput: {
    // Consumes tensor inside subgraph
    return t->id == getTensorIds().second;
  }
  case OpStatus::Normal: {
    return getOp()->input->contains(t);
  }
  case OpStatus::Enter: {
    auto calledGraphs = op->getCalledGraphs();
    auto calledGraph  = calledGraphs.at(getSubgraphIndex());

    if (getOp()->input->contains(t)) {
      auto indices = getOp()->input->indices(t);
      for (InIndex opInIndex : indices) {
        InIndex sgInIndex =
            op->opInToSubgraphInIndex(getSubgraphIndex(), opInIndex);

        if (sgInIndex > -1 && sgInIndex < calledGraph->getInputIds().size() &&
            calledGraph->getTensors()
                .get(calledGraph->getInputId(sgInIndex))
                ->isLoopTripCounter()) {
          // Loop trip count is not copied
          return true;
        }

        if (sgInIndex < 0 || calledGraph->getInputIds().size()) {
          // Is an op input, but not a subgraph input
          return true;
        }
      }
    }
    return false;
  }
  case OpStatus::Exit: {
    auto calledGraphs = op->getCalledGraphs();
    auto calledGraph  = calledGraphs.at(getSubgraphIndex());

    // Consumes tensor inside subgraph (if no CopyOutput exists for this output)
    auto outputIds      = calledGraph->getOutputIds();
    auto it             = std::find(outputIds.begin(), outputIds.end(), t->id);
    OutIndex opOutIndex = op->subgraphOutToOpOutIndex(
        getSubgraphIndex(), std::distance(outputIds.begin(), it));
    if (it != outputIds.end() && !op->output->hasIndex(opOutIndex)) {
      // Is a subgraph output, but not an op output
      return true;
    }

    return false;
  }
  default:
    return false;
  }
}

// If the operation overwrites t
bool LivenessNode::overwritesTensor(Tensor *t) const {
  auto op = getOp();

  switch (status) {
  case OpStatus::CopyInput: {
    return isProducerOf(t);
  }
  case OpStatus::CopyLoopCarried: {
    return isProducerOf(t);
  }
  case OpStatus::CopyModified: {
    return isProducerOf(t);
  }
  case OpStatus::CopyOutput: {
    return isProducerOf(t);
  }
  case OpStatus::Normal: {
    return op->overwritesTensor(t);
  }
  case OpStatus::Enter: {
    return isProducerOf(t);
  }
  default: {
    return false;
  }
  }
}

// If the operation modifies t
bool LivenessNode::modifiesTensor(Tensor *t) const {
  auto op = getOp();

  switch (status) {
  case OpStatus::CopyInput: {
    return isProducerOf(t);
  }
  case OpStatus::CopyLoopCarried: {
    return isProducerOf(t);
  }
  case OpStatus::CopyModified: {
    return isProducerOf(t);
  }
  case OpStatus::CopyOutput: {
    return isProducerOf(t);
  }
  case OpStatus::Normal: {
    return op->modifiesTensor(t);
  }
  case OpStatus::Enter: {
    return isProducerOf(t);
  }
  default: {
    return false;
  }
  }
}

LivenessAnalyzer::LivenessAnalyzer(
    const Ir *ir_,
    const SubgraphCopyingStrategy *subgraphCopyingStrat_)
    : ir(ir_), subgraphCopyingStrat(subgraphCopyingStrat_) {}

void LivenessAnalyzer::apply() {

  // Global schedule including all subgraphs recursively
  graphCallSiteOps[ir->getMainGraph().id] = {};

  for (const Graph *sgraph : ir->getAllGraphs()) {
    graphOpSchedule[sgraph->id] =
        sgraph->getOpSchedule({}, RequireOptimalSchedule::Yes);
  }

  PendingCopies pendingCopies;
  addToSchedule(&(ir->getMainGraph()), false, {}, pendingCopies);

  if (!pendingCopies.empty()) {
    throw error("{} pending copies were not added, IR is inconsistent",
                pendingCopies.size());
  }

  for (int64_t i = 0; i < opSchedule.size(); ++i) {
    for (TensorId tensorId : opSchedule.at(i).usedTensorIds()) {
      tensorScheduleMap[tensorId].push_back(i);
    }
  }

  // Log the global schedule.
  logging::devicex::trace("[LivenessAnalyzer] Global schedule:");
  nonstd::optional<ExecutionContext> context;
  for (size_t i = 0; i < opSchedule.size(); ++i) {
    const auto &node = opSchedule.at(i);
    auto currentContext =
        sanitizeExecutionContext(node.getOp()->settings.executionContext);
    if (!context.has_value() || currentContext != context) {
      if (context.has_value()) {
        contextEnds[context.value()] = std::max<size_t>(0, i - 1);
      }
      contextStarts[currentContext] = std::min<size_t>(opSchedule.size(), i);
      context                       = currentContext;
    }
    logging::devicex::trace(
        "[LivenessAnalyzer] #{}: {} context: {}", i, node, currentContext);
  }
  if (context.has_value()) {
    contextEnds[context.value()] = std::max<size_t>(0, opSchedule.size() - 1);
  }

  // Log context starts/ends
  logging::devicex::trace("[LivenessAnalyzer] Context starts/ends:");
  for (ExecutionContext context : std::vector<ExecutionContext>{
           ExecutionContext::WeightsFromHostFragment,
           ExecutionContext::OptimizerFromHostFragment,
           ExecutionContext::Normal,
           ExecutionContext::WeightsToHostFragment}) {
    logging::trace("[LivenessAnalyzer] Context: {}, start: {}, end: {}",
                   context,
                   getContextStartIndex(context),
                   getContextEndIndex(context));
  }
}

// Get the start position of a context
int64_t LivenessAnalyzer::getContextStartIndex(ExecutionContext context) const {
  // Treats `Subgraph` and `Normal` the same
  auto it = contextStarts.find(sanitizeExecutionContext(context));
  return it == contextStarts.end() ? -1 : it->second;
}

// Get the end position of a context
int64_t LivenessAnalyzer::getContextEndIndex(ExecutionContext context) const {
  // Treats `Subgraph` and `Normal` the same
  auto it = contextEnds.find(sanitizeExecutionContext(context));
  return it == contextEnds.end() ? -1 : it->second;
}

void LivenessAnalyzer::addCopiesToPending(const Graph *subgraph,
                                          bool isDuplicate,
                                          const CallStack &callStack,
                                          SubgraphIndex subgraphIndex,
                                          PendingCopies &pendingCopies) {
  assert(!callStack.empty());
  Op *op = callStack.back();

  // Get all required input copies.
  for (auto input : op->input->tensorIdMap()) {
    auto sgInIndex = op->opInToSubgraphInIndex(subgraphIndex, input.first);
    if (sgInIndex > -1 && sgInIndex < subgraph->getInputIds().size()) {
      // Copy callsite's input tensor to subgraph's input tensor.

      Tensor *sgTensor =
          subgraph->getTensors().get(subgraph->getInputIds().at(sgInIndex));
      // Loop trip count is not copied
      if (!sgTensor->isLoopTripCounter()) {
        pendingCopies.push_back({callStack,
                                 OpStatus::CopyInput,
                                 input.first,
                                 subgraphIndex,
                                 isDuplicate});
      }
    }
  }

  // Get all required output copies.
  for (auto output : op->output->tensorIdMap()) {
    auto sgOutIndex = op->opOutToSubgraphOutIndex(subgraphIndex, output.first);
    if (sgOutIndex > -1 && sgOutIndex < subgraph->getOutputIds().size()) {
      // Copy subgraph's output tensor to callsite's output tensor.
      pendingCopies.push_back({callStack,
                               OpStatus::CopyOutput,
                               output.first,
                               subgraphIndex,
                               isDuplicate});
    }
  }

  // Get all input tensors that were modified (and hence need to be copied).
  for (auto input : op->input->tensorIdMap()) {
    auto sgInIndex = op->opInToSubgraphInIndex(subgraphIndex, input.first);
    if (sgInIndex >= 0) {
      Tensor *sgTensor =
          subgraph->getTensors().get(subgraph->getInputIds().at(sgInIndex));
      // Loop trip count is not copied
      if (!sgTensor->isLoopTripCounter()) {
        // Check for subgraph modified input
        auto modifiedRegions = op->modifies(input.first);
        if (std::any_of(modifiedRegions.begin(),
                        modifiedRegions.end(),
                        [](const view::Region &r) { return !r.isEmpty(); })) {
          // Copy modified subgraph inputs to the callsite.
          pendingCopies.push_back({callStack,
                                   OpStatus::CopyModified,
                                   input.first,
                                   subgraphIndex,
                                   isDuplicate});
        }
      }
    }
  }
}

void LivenessAnalyzer::expandSubgraph(const Graph *subgraph,
                                      bool isDuplicate,
                                      const CallStack &callStack,
                                      SubgraphIndex subgraphIndex,
                                      PendingCopies &pendingCopies) {

  assert(!callStack.empty());
  Op *op = callStack.back();

  // Add copy notes to pendingCopies.
  addCopiesToPending(
      subgraph, isDuplicate, callStack, subgraphIndex, pendingCopies);

  // Add enter node to the stack.
  int64_t enterLocation = addNodeToSchedule(
      {callStack, OpStatus::Enter, 0, subgraphIndex, isDuplicate},
      pendingCopies);

  auto &callSites = graphCallSiteOps[subgraph->id];
  if (std::find(callSites.begin(), callSites.end(), op) == callSites.end()) {
    callSites.push_back(op);
  }

  if (dynamic_cast<LoopOp *>(op)) {
    // Add the loop subgraph twice (iteration 0 and 1),
    // so that we can reason about loop-carried
    // dependencies (via inductive proof) properly
    for (int64_t iteration = 0; iteration < 2; ++iteration) {
      auto bodyOutputIds = subgraph->getOutputIds();
      // Loop carried dependencies between iteration -1, 0 and 1
      // (see loop.hpp)
      for (int i = 0; i < bodyOutputIds.size(); ++i) {
        addNodeToSchedule({callStack,
                           OpStatus::CopyLoopCarried,
                           i,
                           subgraphIndex,
                           isDuplicate || iteration > 0},
                          pendingCopies);
      }
      // Loop iteration 0 & 1
      addToSchedule(
          subgraph, isDuplicate || iteration > 0, callStack, pendingCopies);
    }
  } else {
    // Add ops in the subgraph once.
    addToSchedule(subgraph, isDuplicate, callStack, pendingCopies);
  }

  // Add exit nodex.
  addNodeToSchedule({callStack, OpStatus::Exit, 0, subgraphIndex, isDuplicate},
                    pendingCopies);

  callSiteLinks[enterLocation].push_back(opSchedule.size() - 1);
  callSiteLinksInv[opSchedule.size() - 1].push_back(enterLocation);
}

int64_t LivenessAnalyzer::addNodeToSchedule(const LivenessNode &nodeToAdd,
                                            PendingCopies &pendingCopies) {

  // Insert copies that our strategy suggests should come before the node.
  auto before = subgraphCopyingStrat->getIndicesOfCopiesToInsertBeforeNode(
      nodeToAdd, pendingCopies);
  processSubgraphCopyingStrategyIndices(pendingCopies, before);

  // Remember the position where we add nodeToAdd.
  int64_t nodeToAddPosition = opSchedule.size();
  opSchedule.push_back(nodeToAdd);

  // Insert copies that our strategy suggests should come after the node.
  auto after = subgraphCopyingStrat->getIndicesOfCopiesToInsertAfterNode(
      nodeToAdd, pendingCopies);
  processSubgraphCopyingStrategyIndices(pendingCopies, after);

  // Return the position where the op was added.
  return nodeToAddPosition;
}

void LivenessAnalyzer::processSubgraphCopyingStrategyIndices(
    PendingCopies &pendingCopies,
    std::vector<size_t> &chosenIndices) {

  // Add the chosen indices to the schedule (order is important).
  for (auto index : chosenIndices) {
    opSchedule.push_back(pendingCopies.at(index));
  }

  // Remove all nodes in the indices list from pendingCopies. Achieve this
  // by constructing a new pendingCopies list and copying nodes accross. The
  // alternative would be to erase nodes from the existing list but this may
  // be less efficient.
  PendingCopies newPendingCopies;
  newPendingCopies.reserve(pendingCopies.size() - chosenIndices.size());

  // Work out which indices we're keeping.
  std::vector<bool> keepNodeAtIndex(pendingCopies.size(), true);
  for (auto index : chosenIndices) {
    keepNodeAtIndex.at(index) = false;
  }

  // Populate newPendingCopies with nodes we are keeping.
  for (size_t i = 0; i < pendingCopies.size(); ++i) {
    if (keepNodeAtIndex.at(i)) {
      newPendingCopies.push_back(pendingCopies[i]);
    }
  }

  // Move newPendingCopies to pendingCopies.
  pendingCopies = std::move(newPendingCopies);
}

void LivenessAnalyzer::addToSchedule(const Graph *graphToAdd,
                                     bool isDuplicate,
                                     CallStack callStack,
                                     PendingCopies &pendingCopies) {

  auto &schedule = graphOpSchedule.at(graphToAdd->id);
  for (Op *op : schedule) {
    logging::trace("[LivenessAnalyzer] Adding Op {} to schedule.",
                   op->debugName());
    auto newCallStack = callStack;
    newCallStack.push_back(op);

    // Expand subgraphs, if any.
    auto subgraphs = op->getCalledGraphs();
    if (!subgraphs.empty()) {
      // This op has subgraphs, expand them.
      for (SubgraphIndex g = 0; g < subgraphs.size(); ++g) {
        // Work out enter location.
        int64_t enterLocation = opSchedule.size();
        opScheduleMap[op].push_back(enterLocation);
        // Expand subgraph.
        expandSubgraph(
            subgraphs[g], isDuplicate, newCallStack, g, pendingCopies);
      }
    } else {
      // Work out enter location.
      int64_t enterLocation = opSchedule.size();
      opScheduleMap[op].push_back(enterLocation);
      // This op has no subgraphs.
      addNodeToSchedule({newCallStack, OpStatus::Normal, 0, 0, isDuplicate},
                        pendingCopies);
    }
  }
}

int64_t LivenessAnalyzer::getGlobalSchedulePosition(CallStack ops) const {
  int64_t index = -1;
  for (Op *op : ops) {
    for (int64_t i : opScheduleMap.at(op)) {
      if (i > index) {
        // First occurence of op after index
        index = i;
        break;
      }
    }
  }
  return index;
}

std::ostream &operator<<(std::ostream &os, const OpStatus &opStatus) {
  switch (opStatus) {
  case OpStatus::Normal:
    os << "Normal";
    break;
  case OpStatus::Enter:
    os << "Enter";
    break;
  case OpStatus::CopyInput:
    os << "CopyInput";
    break;
  case OpStatus::CopyLoopCarried:
    os << "CopyLoopCarried";
    break;
  case OpStatus::CopyOutput:
    os << "CopyOutput";
    break;
  case OpStatus::CopyModified:
    os << "CopyModified";
    break;
  case OpStatus::Exit:
    os << "Exit";
    break;
  default:
    os << "Undefined";
    break;
  }
  return os;
}

std::ostream &operator<<(std::ostream &os, const LivenessNode &livenessNode) {
  // Output callstack as "[main>sg0>sg1]".
  bool isFirst = true;
  os << "[";
  for (auto &op : livenessNode.getCallStack()) {
    if (!isFirst)
      os << ">";
    os << op->getGraph().id.str();
    isFirst = false;
  }
  os << "] ";
  os << livenessNode.getOp()->debugName() << " ";
  os << livenessNode.getStatus() << " ";
  os << livenessNode.getIndex() << " ";
  os << livenessNode.getSubgraphIndex() << " ";
  if (livenessNode.getDuplicate())
    os << livenessNode.getSubgraphIndex() << " (duplicate)";
  os << "{" << livenessNode.getTensorIds().first << ", "
     << livenessNode.getTensorIds().second << "}";
  return os;
}

} // namespace liveness
} // namespace popart
