// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <functional>

#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/liveness.hpp>
#include <popart/op/if.hpp>
#include <popart/op/loop.hpp>
#include <popart/op/subgraph.hpp>
#include <popart/scheduler_requireoptimal.hpp>

namespace popart {
namespace liveness {

LivenessNode::LivenessNode(const OpStatus status_,
                           int index_,
                           SubgraphIndex subgraphIndex_,
                           bool isDuplicate_)
    : callStack({}), status(status_), index(index_),
      subgraphIndex(subgraphIndex_), isDuplicate(isDuplicate_) {}

LivenessNode::LivenessNode(std::vector<Op *> callStack_,
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
      if (sgInIndex < 0 || sgInIndex >= calledGraph->getInputIds().size()) {
        // Op input which is not a subgraph input
        usedIds.insert(input.second);
      }
    }

    auto inputIds = calledGraph->getInputIds();
    for (int64_t i = 0; i < inputIds.size(); ++i) {
      auto opInIndex = op->subgraphInToOpInIndex(getSubgraphIndex(), i);
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
    auto inputIds     = calledGraph->getInputIds();
    auto it           = std::find(inputIds.begin(), inputIds.end(), t->id);
    InIndex opInIndex = op->subgraphInToOpInIndex(
        getSubgraphIndex(), std::distance(inputIds.begin(), it));
    if (it != inputIds.end() && !op->hasInput(opInIndex)) {
      // Is a subgraph input, but not an op input
      return true;
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

LivenessAnalyzer::LivenessAnalyzer(const Ir *ir_) : ir(ir_) {}

void LivenessAnalyzer::apply() {

  // Global schedule including all subgraphs recursively
  graphCallSiteOps[ir->getMainGraph().id] = {};

  for (const Graph *sgraph : ir->getAllGraphs()) {
    graphOpSchedule[sgraph->id] =
        sgraph->getOpSchedule({}, RequireOptimalSchedule::Yes);
  }
  addToSchedule(&(ir->getMainGraph()), false, {});

  for (int64_t i = 0; i < opSchedule.size(); ++i) {
    for (TensorId tensorId : opSchedule.at(i).usedTensorIds()) {
      tensorScheduleMap[tensorId].push_back(i);
    }
  }

  // Log the global schedule.
  logging::devicex::debug("[LivenessAnalyzer] Global schedule:");
  size_t i = 0;
  for (const auto &node : opSchedule) {
    logging::devicex::debug("[LivenessAnalyzer] #{}: {}", i++, node);
  }
}

void LivenessAnalyzer::expandSubgraph(const Graph *graphToAdd,
                                      const Graph *subgraph,
                                      int64_t enterLocation,
                                      bool isDuplicate,
                                      std::vector<Op *> callStack,
                                      SubgraphIndex subgraphIndex) {

  assert(!callStack.empty());
  Op *op = callStack.back();

  opSchedule.emplace_back(
      callStack, OpStatus::Enter, 0, subgraphIndex, isDuplicate);
  for (auto input : op->input->tensorIdMap()) {
    auto sgInIndex = op->opInToSubgraphInIndex(subgraphIndex, input.first);
    if (sgInIndex >= 0) {
      opSchedule.push_back({callStack,
                            OpStatus::CopyInput,
                            input.first,
                            subgraphIndex,
                            isDuplicate});
    }
  }

  auto &callSites = graphCallSiteOps[subgraph->id];
  if (std::find(callSites.begin(), callSites.end(), op) == callSites.end()) {
    callSites.push_back(op);
  }

  // Add the loop subgraph twice (iteration 0 and 1),
  // so that we can reason about loop-carried
  // dependencies (via inductive proof) properly
  if (dynamic_cast<LoopOp *>(op)) {
    for (int64_t iteration = 0; iteration < 2; ++iteration) {
      auto bodyOutputIds = subgraph->getOutputIds();
      // Loop carried dependencies between iteration -1, 0 and 1
      // (see loop.hpp)
      for (int64_t i = 0; i < bodyOutputIds.size(); ++i) {
        opSchedule.emplace_back(callStack,
                                OpStatus::CopyLoopCarried,
                                i,
                                subgraphIndex,
                                isDuplicate || iteration > 0);
      }
      // Loop iteration 0 & 1
      addToSchedule(subgraph, isDuplicate || iteration > 0, callStack);
    }
  } else {
    addToSchedule(subgraph, isDuplicate, callStack);
  }

  // Insert the "exit" locations of subgraphs into the schedule
  for (auto output : op->output->tensorIdMap()) {
    auto sgOutIndex = op->opOutToSubgraphOutIndex(subgraphIndex, output.first);
    if (sgOutIndex > -1 && sgOutIndex < subgraph->getOutputIds().size()) {
      // Copy subgraph outputs to the main graph
      opSchedule.emplace_back(callStack,
                              OpStatus::CopyOutput,
                              output.first,
                              subgraphIndex,
                              isDuplicate);
    }
  }
  for (auto input : op->input->tensorIdMap()) {
    auto sgInIndex = op->opInToSubgraphInIndex(subgraphIndex, input.first);
    if (sgInIndex >= 0) {
      // Check for subgraph modified input
      auto modifiedRegions = op->modifies(input.first);
      if (std::any_of(modifiedRegions.begin(),
                      modifiedRegions.end(),
                      [](const view::Region &r) { return !r.isEmpty(); })) {
        // Copy modified subgraph inputs to the main graph
        opSchedule.push_back({callStack,
                              OpStatus::CopyModified,
                              input.first,
                              subgraphIndex,
                              isDuplicate});
      }
    }
  }

  opSchedule.emplace_back(
      callStack, OpStatus::Exit, 0, subgraphIndex, isDuplicate);
  callSiteLinks[enterLocation].push_back(opSchedule.size() - 1);
}

void LivenessAnalyzer::addToSchedule(const Graph *graphToAdd,
                                     bool isDuplicate,
                                     std::vector<Op *> callStack) {
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
        expandSubgraph(graphToAdd,
                       subgraphs[g],
                       enterLocation,
                       isDuplicate,
                       newCallStack,
                       g);
      }
    } else {
      // Work out enter location.
      int64_t enterLocation = opSchedule.size();
      opScheduleMap[op].push_back(enterLocation);
      // This op has no subgraphs.
      opSchedule.emplace_back(
          newCallStack, OpStatus::Normal, 0, 0, isDuplicate);
    }
  }
}

int64_t
LivenessAnalyzer::getGlobalSchedulePosition(std::vector<Op *> ops) const {
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
