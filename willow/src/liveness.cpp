// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/liveness.hpp>

namespace popart {
namespace liveness {

LivenessAnalyzer::LivenessAnalyzer(const Ir *ir_) : ir(ir_){};

void LivenessAnalyzer::apply() {
  // Global schedule including all subgraphs recursively
  graphCallSiteOps[ir->getMainGraph().id] = {};

  for (const Graph *sgraph : ir->getAllGraphs()) {
    graphOpSchedule[sgraph->id] = sgraph->getOpSchedule({});
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
      opSchedule.push_back({current, OpStatus::Normal, 0});
    } else {
      opSchedule.push_back({current, OpStatus::Enter, 0});
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
        opSchedule.push_back({current, OpStatus::CopyOutput, i});
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
      opSchedule.push_back({current, OpStatus::Exit, 0});
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
