// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/liveness.hpp>
#include <popart/op/subgraph.hpp>
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
  case OpStatus::CopyModified: {
    SubgraphOp *sgOp = static_cast<SubgraphOp *>(getOp());
    auto sgInIdx     = sgOp->opInToSubgraphInIndex(index);
    if (sgInIdx > -1 && sgInIdx < sgOp->getCalledGraph().getInputIds().size()) {
      return {getOp()->inId(index), sgOp->getCalledGraph().getInputId(sgInIdx)};
    } else {
      return {"", ""};
    }
  } break;
  case OpStatus::CopyOutput: {
    SubgraphOp *sgOp = static_cast<SubgraphOp *>(callStack.back());
    auto sgOutIdx    = sgOp->opOutToSubgraphOutIndex(index);
    if (sgOutIdx > -1 &&
        sgOutIdx < sgOp->getCalledGraph().getOutputIds().size()) {
      return {getOp()->outId(index),
              sgOp->getCalledGraph().getOutputId(sgOutIdx)};
    } else {
      return {"", ""};
    }
  } break;
  case OpStatus::Normal:
  case OpStatus::Enter:
  case OpStatus::Exit:
  default:
    return {"", ""};
  }
}

bool LivenessNode::isProducerOf(Tensor *t) const {
  switch (status) {
  case OpStatus::CopyInput: {
    // Produces tensor inside subgraph
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
    logging::trace("[LivenessAnalyzer] Adding Op {} to schedule.",
                   op->debugName());
    auto current = callStack;
    current.push_back(op);

    // Insert the actual op ("enter" location for subgraphing ops)

    int64_t enter_location = 0;

    enter_location = opSchedule.size();
    opScheduleMap[op].push_back(enter_location);

    if (SubgraphOp *sgOp = dynamic_cast<SubgraphOp *>(op)) {
      opSchedule.emplace_back(current, OpStatus::Enter, 0);
      for (auto input : sgOp->input->tensorIdMap()) {
        auto sgInIndex = sgOp->opInToSubgraphInIndex(input.first);
        if (sgInIndex > -1 &&
            sgInIndex < sgOp->getCalledGraph().getInputIds().size()) {
          opSchedule.push_back({current, OpStatus::CopyInput, input.first});
        }
      }
    } else {
      opSchedule.emplace_back(current, OpStatus::Normal, 0);
    }

    // Inspect subgraphs
    if (SubgraphOp *sgOp = dynamic_cast<SubgraphOp *>(op)) {
      auto &subgraph  = sgOp->getCalledGraph();
      auto &callSites = graphCallSiteOps[subgraph.id];
      if (std::find(callSites.begin(), callSites.end(), op) ==
          callSites.end()) {
        callSites.push_back(op);
      }
      addToSchedule(&subgraph, current);
      // Insert the "exit" locations of subgraphs into the schedule
      for (auto output : sgOp->output->tensorIdMap()) {
        auto sgOutIndex = sgOp->opOutToSubgraphOutIndex(output.first);
        if (sgOutIndex > -1 &&
            sgOutIndex < sgOp->getCalledGraph().getOutputIds().size()) {
          opSchedule.emplace_back(current, OpStatus::CopyOutput, output.first);
        }
      }
      for (auto input : sgOp->input->tensorIdMap()) {
        auto sgInIndex = sgOp->opInToSubgraphInIndex(input.first);
        if (sgInIndex > -1 &&
            sgInIndex < sgOp->getCalledGraph().getInputIds().size()) {
          // Check for subgraph modified input
          auto modifiedRegions = op->modifies(input.first);
          if (std::any_of(modifiedRegions.begin(),
                          modifiedRegions.end(),
                          [](const view::Region &r) { return !r.isEmpty(); })) {
            opSchedule.push_back(
                {current, OpStatus::CopyModified, input.first});
          }
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
