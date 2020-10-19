// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/liveness.hpp>
#include <popart/scheduler_requireoptimal.hpp>

namespace popart {
namespace liveness {

LivenessNode::LivenessNode(std::vector<Op *> callStack_,
                           const OpStatus status_,
                           int index_)
    : callStack(callStack_), status(status_), index(index_) {}

std::pair<TensorId, TensorId> LivenessNode::getTensorIds() const {
  switch (status) {
  case OpStatus::CopyInput:
  case OpStatus::CopyModified:
    return {getOp()->inId(index),
            getOp()->getCalledGraphs().front()->getInputId(index)};
  case OpStatus::CopyOutput:
    return {getOp()->outId(index),
            getOp()->getCalledGraphs().front()->getOutputId(index)};
  case OpStatus::Normal:
  case OpStatus::Enter:
  case OpStatus::Exit:
  default:
    return {"", ""};
  }
}

bool LivenessNode::isProducerOf(Tensor *t) const {
  switch (status) {
  case OpStatus::CopyInput:
    // Produces tensor inside subgraph
    return t->id == getTensorIds().second;
  case OpStatus::CopyModified:
  case OpStatus::CopyOutput:
    // Produces tensor outside subgraph
    return t->id == getTensorIds().first;
  case OpStatus::Normal:
    return t->hasProducer() && t->getProducer() == getOp();
  case OpStatus::Enter:
  case OpStatus::Exit:
  default:
    return false;
  }
}

bool LivenessNode::isConsumerOf(Tensor *t) const {
  switch (status) {
  case OpStatus::CopyInput:
    // Consumes tensor outside subgraph
    return t->id == getTensorIds().first;
  case OpStatus::CopyModified:
    // Consumes tensor inside subgraph
    return t->id == getTensorIds().second;
  case OpStatus::CopyOutput:
    // Consumes tensor inside subgraph
    return t->id == getTensorIds().second;
  case OpStatus::Normal:
    return getOp()->input->contains(t);
  case OpStatus::Enter:
  case OpStatus::Exit:
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
  addToSchedule(&(ir->getMainGraph()), {});
}

void LivenessAnalyzer::addToSchedule(const Graph *graphToAdd,
                                     std::vector<Op *> callStack) {
  auto &schedule = graphOpSchedule.at(graphToAdd->id);
  for (Op *op : schedule) {
    auto current = callStack;
    current.push_back(op);

    auto const &calledGraphs = op->getCalledGraphs();

    // Insert the actual op ("enter" location for subgraphing ops)

    int64_t enter_location = 0;

    enter_location = opSchedule.size();
    opScheduleMap[op].push_back(enter_location);

    if (calledGraphs.empty()) {
      opSchedule.emplace_back(current, OpStatus::Normal, 0);
    } else {
      opSchedule.emplace_back(current, OpStatus::Enter, 0);
      for (const Graph *subgraph : calledGraphs) {
        for (InIndex i = 0; i < subgraph->getInputIds().size(); ++i) {
          opSchedule.push_back({current, OpStatus::CopyInput, i});
        }
      }
    }

    // Inspect subgraphs
    for (const Graph *subgraph : calledGraphs) {
      auto &callSites = graphCallSiteOps[subgraph->id];
      if (std::find(callSites.begin(), callSites.end(), op) ==
          callSites.end()) {
        callSites.push_back(op);
      }
      addToSchedule(subgraph, current);
      // Insert the "exit" locations of subgraphs into the schedule
      for (OutIndex i = 0; i < subgraph->getOutputIds().size(); ++i) {
        opSchedule.emplace_back(current, OpStatus::CopyOutput, i);
      }
      for (OutIndex i = 0; i < subgraph->getInputIds().size(); ++i) {
        // Check for subgraph modified input
        auto modifiedRegions = current.back()->modifies(i);
        if (std::any_of(modifiedRegions.begin(),
                        modifiedRegions.end(),
                        [](const view::Region &r) { return !r.isEmpty(); })) {
          opSchedule.push_back({current, OpStatus::CopyModified, i});
        }
      }
      opSchedule.emplace_back(current, OpStatus::Exit, 0);
      callSiteLinks[enter_location].push_back(opSchedule.size() - 1);
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

} // namespace liveness
} // namespace popart
