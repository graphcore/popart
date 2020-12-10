// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/liveness.hpp>
#include <popart/op/loop.hpp>
#include <popart/op/subgraph.hpp>
#include <popart/scheduler_requireoptimal.hpp>

namespace popart {
namespace liveness {

LivenessNode::LivenessNode(const OpStatus status_, int index_)
    : callStack(), status(status_), index(index_) {}

LivenessNode::LivenessNode(std::vector<Op *> callStack_,
                           const OpStatus status_,
                           int index_)
    : callStack(callStack_), status(status_), index(index_) {}

std::set<TensorId> LivenessNode::usedTensorIds() const {
  std::set<TensorId> used;

  switch (status) {
  case OpStatus::CopyInput:
  case OpStatus::CopyLoopCarried:
  case OpStatus::CopyModified:
  case OpStatus::CopyOutput: {
    used.insert(getTensorIds().first);
    used.insert(getTensorIds().second);
    break;
  }
  case OpStatus::Normal: {
    for (auto &input : getOp()->input->tensorIdMap()) {
      used.insert(input.second);
    }
    for (auto &output : getOp()->output->tensorIdMap()) {
      used.insert(output.second);
    }
    break;
  }
  case OpStatus::Enter: {
    if (SubgraphOp *subgraphOp = dynamic_cast<SubgraphOp *>(getOp())) {
      for (auto &input : subgraphOp->input->tensorIdMap()) {
        auto sgInIndex = subgraphOp->opInToSubgraphInIndex(input.first);
        if (sgInIndex < 0 ||
            sgInIndex >= subgraphOp->getCalledGraph().getInputIds().size()) {
          // Op input which is not a subgraph input
          used.insert(input.second);
        }
      }

      auto inputIds = subgraphOp->getCalledGraph().getInputIds();
      for (int64_t i = 0; i < inputIds.size(); ++i) {
        auto opInIndex = subgraphOp->subgraphInToOpInIndex(i);
        if (!subgraphOp->hasInput(opInIndex)) {
          // Subgraph input which is not an op input
          used.insert(inputIds.at(i));
        }
      }
    }
    break;
  }
  case OpStatus::Exit: {
    if (SubgraphOp *subgraphOp = dynamic_cast<SubgraphOp *>(getOp())) {
      for (auto &output : subgraphOp->output->tensorIdMap()) {
        auto sgOutIndex = subgraphOp->opOutToSubgraphOutIndex(output.first);
        if (sgOutIndex < 0 ||
            sgOutIndex >= subgraphOp->getCalledGraph().getOutputIds().size()) {
          // Op output which is not a subgraph output
          used.insert(output.second);
        }
      }

      auto outputIds = subgraphOp->getCalledGraph().getInputIds();
      for (int64_t i = 0; i < outputIds.size(); ++i) {
        auto opOutIndex = subgraphOp->subgraphOutToOpOutIndex(i);
        if (!subgraphOp->output->hasIndex(opOutIndex)) {
          // Subgraph output which is not an op output
          used.insert(outputIds.at(i));
        }
      }
    }
    break;
  }
  default:
    break;
  }

  return used;
}

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
  case OpStatus::CopyLoopCarried: {
    SubgraphOp *sgOp = static_cast<SubgraphOp *>(callStack.back());
    // Loop carried dependencies
    auto sgOutIdx = index;
    auto sgInIdx  = index + 1;
    return {sgOp->getCalledGraph().getOutputId(sgOutIdx),
            sgOp->getCalledGraph().getInputId(sgInIdx)};
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
    // Produces tensor inside subgraph (if no CopyInput exists for this input)
    if (SubgraphOp *subgraphOp = dynamic_cast<SubgraphOp *>(getOp())) {
      auto inputIds     = subgraphOp->getCalledGraph().getInputIds();
      auto it           = std::find(inputIds.begin(), inputIds.end(), t->id);
      InIndex opInIndex = subgraphOp->subgraphInToOpInIndex(
          std::distance(inputIds.begin(), it));
      if (it != inputIds.end() && !subgraphOp->hasInput(opInIndex)) {
        // Is a subgraph input, but not an op input
        return true;
      }
    }
    return false;
  }
  case OpStatus::Exit: {
    // Produces tensor outside subgraph (no CopyOutput)
    if (SubgraphOp *subgraphOp = dynamic_cast<SubgraphOp *>(getOp())) {
      if (t->hasProducer() && t->getProducer() == getOp()) {
        OutIndex opOutIndex = getOp()->output->indices(t).front();
        OutIndex sgOutIndex = subgraphOp->opOutToSubgraphOutIndex(opOutIndex);
        if (sgOutIndex < 0 ||
            sgOutIndex >= subgraphOp->getCalledGraph().getOutputIds().size()) {
          // Is an op output, but not a subgraph output
          return true;
        }
      }
    }
    return false;
  }
  default:
    return false;
  }
}

bool LivenessNode::isConsumerOf(Tensor *t) const {
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
    if (getOp()->input->contains(t)) {
      if (SubgraphOp *subgraphOp = dynamic_cast<SubgraphOp *>(getOp())) {
        auto indices = getOp()->input->indices(t);
        for (InIndex opInIndex : indices) {
          InIndex sgInIndex = subgraphOp->opInToSubgraphInIndex(opInIndex);
          if (sgInIndex < 0 ||
              sgInIndex >= subgraphOp->getCalledGraph().getInputIds().size()) {
            // Is an op input, but not a subgraph input
            return true;
          }
        }
      }
    }
    return false;
  }
  case OpStatus::Exit: {
    // Consumes tensor inside subgraph (if no CopyOutput exists for this output)
    if (SubgraphOp *subgraphOp = dynamic_cast<SubgraphOp *>(getOp())) {
      auto outputIds = subgraphOp->getCalledGraph().getOutputIds();
      auto it        = std::find(outputIds.begin(), outputIds.end(), t->id);
      OutIndex opOutIndex = subgraphOp->subgraphOutToOpOutIndex(
          std::distance(outputIds.begin(), it));
      if (it != outputIds.end() && !subgraphOp->output->hasIndex(opOutIndex)) {
        // Is a subgraph output, but not an op output
        return true;
      }
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
  addToSchedule(&(ir->getMainGraph()), {});

  for (int64_t i = 0; i < opSchedule.size(); ++i) {
    for (TensorId tensorId : opSchedule.at(i).usedTensorIds()) {
      tensorScheduleMap[tensorId].push_back(i);
    }
  }
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

      // Add the loop subgraph twice (iteration 0 and 1),
      // so that we can reason about loop-carried
      // dependencies (via inductive proof) properly
      if (op->isConvertibleTo<LoopOp>()) {
        auto bodyOutputIds = subgraph.getOutputIds();
        // Loop carried dependencies between iteration 0 and 1
        // (see loop.hpp)
        for (int64_t i = 0; i < bodyOutputIds.size(); ++i) {
          opSchedule.emplace_back(current, OpStatus::CopyLoopCarried, i);
        }
        // Second loop iteration
        addToSchedule(&subgraph, current);
      }

      // Insert the "exit" locations of subgraphs into the schedule
      for (auto output : sgOp->output->tensorIdMap()) {
        auto sgOutIndex = sgOp->opOutToSubgraphOutIndex(output.first);
        if (sgOutIndex > -1 &&
            sgOutIndex < sgOp->getCalledGraph().getOutputIds().size()) {
          // Copy subgraph outputs to the main graph
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
            // Copy modified subgraph inputs to the main graph
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
  os << livenessNode.getIndex();
  return os;
}

} // namespace liveness
} // namespace popart
