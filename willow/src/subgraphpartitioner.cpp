// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <cassert>

#include <poprithms/logging/timepartitionlogger.hpp>
#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/liveness.hpp>
#include <popart/op.hpp>
#include <popart/op/call.hpp>
#include <popart/op/if.hpp>

#include <popart/subgraphpartitioner.hpp>

namespace popart {
namespace liveness {

namespace {

// Check if LivenessAnalyzer node is a normal op in subgraph.
bool isNormal(const Graph &graph, const LivenessNode &node) {
  return (node.getOp()->getGraph().id == graph.id);
}

// Check if node is a specific status belonging to a CallOp in subgraph.
bool isCallOpWithStatus(const Graph &graph,
                        const LivenessNode &node,
                        OpStatus opStatus) {
  if (node.getOp()->getGraph().id == graph.id) {
    CallOp *callOp = dynamic_cast<CallOp *>(node.getOp());
    return callOp && (node.getStatus() == opStatus);
  } else {
    return false;
  }
}

// Check if op is part of a subgraph that our active call op calls.
bool isCopyCallSubgraphPart(const Graph &graph,
                            const LivenessNode &node,
                            CallOp *activeCallOp) {
  if (node.getOp()->getGraph().id != graph.id) {
    return (activeCallOp &&
            node.getOp()->getGraph().id == activeCallOp->getCalledGraph().id);
  } else {
    return false;
  }
}

bool isParentCopy(const Graph &graph,
                  const LivenessNode &node,
                  size_t callstackSize) {
  if (node.getStatus() == OpStatus::CopyInput ||
      node.getStatus() == OpStatus::CopyOutput ||
      node.getStatus() == OpStatus::CopyModified) {
    // Check if it's from a parent.
    const auto &nodeCallStack = node.getCallStack();
    return (nodeCallStack.size() <= callstackSize);
  } else {
    // Not a copy.
    return false;
  }
}

} // namespace

void SubgraphPartitioner::apply() {

  const auto lifetimeTimer =
      ir->timePartitionLogger().scopedStopwatch("SubGraphPartitioner");
  if (!ir) {
    throw internal_error("[SubgraphPartitioner] Ir not set.");
  }

  // Cache main graph results.
  populateCache(ir->getMainGraph());
  // Cache subgraph results.
  for (const Graph *graph : ir->getAllGraphs()) {
    populateCache(*graph);
  }
}

void SubgraphPartitioner::setIr(const Ir *_ir) { ir = _ir; }

void SubgraphPartitioner::setLivenessAnalyzer(
    const LivenessAnalyzer *_liveness) {
  liveness = _liveness;
}

int SubgraphPartitioner::getNumSubgraphParts(const Graph &graph) const {
  // Get a subgraph schedule.
  const auto &partition = cache.at(graph.id.str());

  if (partition.empty()) {
    // No ops in the partition. We accomodate this as an edge case.
    return 0;
  }

  // Return size based on the subgraph part number of the last op.
  return std::get<1>(partition.back()) + 1;
}

SubgraphPartIndex SubgraphPartitioner::getOpSubgraphPartBegin(Op *op) const {
  // Get the subgraph schedule.
  const auto &partition = cache.at(op->getGraph().id.str());

  // Look for the first item in the schedule that matches op.
  using Tup = std::tuple<Node, SubgraphPartIndex>;
  auto findIt =
      std::find_if(partition.cbegin(), partition.cend(), [&](const Tup &tuple) {
        return (std::get<0>(tuple).op == op);
      });

  // Throw meaningful error when we find no such item.
  if (findIt == partition.cend()) {
    throw internal_error(
        "[SubgraphPartitioner] Unable to find op {} in subgraph partition for "
        "{}.",
        op->debugName(),
        op->getGraph().getGraphString());
  }

  return std::get<1>(*findIt);
}

SubgraphPartIndex SubgraphPartitioner::getOpSubgraphPartEnd(Op *op) const {
  // Get the subgraph schedule.
  const auto &partition = cache.at(op->getGraph().id.str());

  // Look for the last item in the schedule that matches op.
  using Tup   = std::tuple<Node, SubgraphPartIndex>;
  auto findIt = std::find_if(
      partition.crbegin(), partition.crend(), [&](const Tup &tuple) {
        return (std::get<0>(tuple).op == op);
      });

  // Throw meaningful error when we find no such item.
  if (findIt == partition.crend()) {
    throw internal_error(
        "[SubgraphPartitioner] Unable to find op {} in subgraph partition for "
        "{}.",
        op->debugName(),
        op->getGraph().getGraphString());
  }

  return std::get<1>(*findIt) + 1;
}

SubgraphPartitioner::CallOpSchedule
SubgraphPartitioner::getCallOpSchedule(CallOp *callOp) const {

  CallOpSchedule result;

  const auto &partition = cache.at(callOp->getGraph().id.str());
  for (const auto &part : partition) {
    if (std::get<0>(part).op == callOp) {
      result.push_back({std::get<0>(part).callOpPart, std::get<1>(part)});
    }
  }

  return result;
}

bool SubgraphPartitioner::isPartitionable(const Graph &graph) {
  const auto &ir = graph.getIr();
  if (ir.getMainGraph().id == graph.id) {
    return false;
  } else {
    bool isPartitionable = true;
    // Iterate over all ops that call this graph.
    for (const auto callSiteOp : graph.getCallSiteOps()) {
      // If it isn't a call op, the graph isn't partitionable.
      bool isCallOp = callSiteOp->isConvertibleTo<CallOp>();
      if (!isCallOp) {
        isPartitionable = false;
      }
    }
    return isPartitionable;
  }
}

SubgraphPartitioner::SubgraphPartition
SubgraphPartitioner::determineSubgraphPartition(const Graph &graph,
                                                bool partitionable) {

  if (!liveness)
    throw internal_error("[SubgraphPartitioner] LivenessAnalyzer not set.");

  SubgraphPartition result;

  // 1) Ensure we have a SubgraphPartition for each child graph already.
  populateCacheForCalledGraphs(graph);

  // Log what we are doing.
  logging::devicex::trace("[SubgraphPartitioner] Determining subgraph "
                          "partition for {} ({}).",
                          graph.getGraphString(),
                          partitionable ? "partitionable" : "unpartitionable");

  const auto &schedule = liveness->getGraphOpSchedule(graph.id);
  if (graph.id == ir->getMainGraph().id) {

    // It's the main graph, no call sites.
    auto lastIndex = liveness->getOpScheduleSize() - 1;
    auto partition =
        getSubgraphPartitionForInstance(graph, schedule, 0, lastIndex, 0);
    result = finaliseSubgraphPartition(partition);

  } else {

    // It's a subgraph. For each instance of the subgraph in the global schedule
    // work out the sequence of ops, expanding any CallOps. If we have multiple
    // instances, ensure they agree on sequence and combine subgraph partition
    // boundaries.

    bool havePartition = false;
    SubgraphPartitionTmp partition;

    // All the following code does is finding enter/exit indices in the global
    // schedule for our graph. We currently do this by looking for enter
    // nodes that call our graph, but there may be better ways of doing this.

    for (size_t i = 0; i < liveness->getOpScheduleSize(); i++) {
      const auto &node = liveness->getOpScheduleAt(i);

      if (!node.getDuplicate() && node.getStatus() == OpStatus::Enter) {
        assert(node.getSubgraphIndex() <
               node.getOp()->getCalledGraphs().size());
        auto calledGraph =
            node.getOp()->getCalledGraphs()[node.getSubgraphIndex()];

        if (graph.id.str() == calledGraph->id.str()) {

          // It's a subgraph op that calls our graph.
          size_t enter = i;

          // Get exit, for subgraph ops we expect exactly 1 exit.
          const auto &exits = liveness->getCallSiteLinksAt(enter);
          if (exits.size() != 1) {
            throw internal_error("[SubgraphPartitioner] Expected one exit "
                                 "point for call to {} (got {}).",
                                 graph.getGraphString(),
                                 exits.size());
          }

          size_t exit = exits.back();

          // Get a schedule for this instance of the subgraph.
          auto callstackSize = node.getCallStack().size();
          auto newPartition  = getSubgraphPartitionForInstance(
              graph, schedule, enter, exit, callstackSize);

          if (!havePartition) {
            partition     = newPartition;
            havePartition = true;
          } else {
            // Different instances may break at different point, we need to
            // accomodate all breaks.
            partition = mergeSubgraphPartitions(graph, partition, newPartition);
          }
        }
      }
    }

    if (!havePartition) {
      throw internal_error("[SubgraphPartitioner] Unable to determine subgraph "
                           "partition for {}. No subgraph ops found in global "
                           "schedule.",
                           graph.getGraphString());
    }

    result = finaliseSubgraphPartition(partition);
  }

  if (!partitionable && result.size() > 0 && std::get<1>(result.back()) > 0) {
    throw internal_error("[SubgraphPartitioner] Found multiple subgraph parts "
                         "for {} (which is a graph that cannot be "
                         "partitioned). This must be due to a parent "
                         "graph copying inputs or outputs in the middle of "
                         "{}, which cannot be achieved without partitioning.",
                         graph.getGraphString(),
                         graph.getGraphString());
  }

  // Log it.
  logSubgraphPartition(graph, result, partitionable);

  return result;
}

SubgraphPartitioner::SubgraphPartitionTmp
SubgraphPartitioner::getSubgraphPartitionForInstance(
    const Graph &graph,
    const std::vector<Op *> &schedule,
    size_t enter,
    size_t exit,
    size_t callstackSize) {

  // Log what we are doing.
  logging::devicex::trace("[SubgraphPartitioner] Looking to extract subgraph "
                          "partition for {} from global schedule "
                          "(indices {} to {}).",
                          graph.getGraphString(),
                          enter,
                          exit);

  SubgraphPartitionTmp result;
  auto &nodes      = std::get<0>(result);
  auto &boundaries = std::get<1>(result);

  // The CallOp in this subgraph that is active, if any.
  CallOp *activeCallOp = nullptr;

  // Keeps track of subgraph parts that have already been called.
  SubgraphPartIndex finalisedPart = -1;
  // Keeps track of subgraph parts that should be called.
  SubgraphPartIndex discoveredPart = -1;

  // Insert the calls.
  auto syncCalls = [&]() {
    // A CallOp must be active.
    if (!activeCallOp)
      throw internal_error("[SubgraphPartitioner] Unable to add calls to "
                           "subgraph parts as no CallOp is active.");

    auto &calledGraph = activeCallOp->getCalledGraph();
    for (SubgraphPartIndex s = finalisedPart + 1; s <= discoveredPart; ++s) {
      // Check the subgraph part exists.
      if (s >= getNumSubgraphParts(calledGraph)) {
        throw internal_error("[SubgraphPartitioner] The subgraph partition for "
                             "{} (op {}) includes a call to subgraph {}, "
                             "subgraph partition {} but {} only has {} parts.",
                             graph.getGraphString(),
                             activeCallOp->debugName(),
                             calledGraph.getGraphString(),
                             s,
                             calledGraph.getGraphString(),
                             getNumSubgraphParts(calledGraph));
      }
      nodes.push_back(
          {activeCallOp, {CallOpPartType::CallSubgraphPart, 0, 0, s}});
    }
    // Remember we inserted them.
    finalisedPart = discoveredPart;
  };

  for (size_t i = enter; i <= exit; ++i) {
    const auto &node = liveness->getOpScheduleAt(i);
    if (!node.getDuplicate()) {

      Op *op     = node.getOp();
      auto index = node.getIndex();

      if (isCallOpWithStatus(graph, node, OpStatus::Enter)) {

        // It's a call op (starting).
        activeCallOp   = dynamic_cast<CallOp *>(node.getOp());
        finalisedPart  = -1;
        discoveredPart = -1;

      } else if (isCallOpWithStatus(graph, node, OpStatus::Exit)) {

        // It's a call op (ending).
        syncCalls();
        auto &calledGraph = activeCallOp->getCalledGraph();
        if (discoveredPart + 1 != getNumSubgraphParts(calledGraph)) {
          throw internal_error("[SubgraphPartitioner] The graph sequence for "
                               "{} (op {}) comprises {} calls to subgraph "
                               "parts of {} (expected {}).",
                               graph.getGraphString(),
                               activeCallOp->debugName(),
                               discoveredPart + 1,
                               calledGraph.getGraphString(),
                               getNumSubgraphParts(calledGraph));
        }
        activeCallOp = nullptr;

      } else if (isCallOpWithStatus(graph, node, OpStatus::CopyInput)) {

        // A CallOp copying input.
        syncCalls();
        nodes.push_back({op, {CallOpPartType::CopyInput, index, 0, 0}});

        logging::devicex::trace("[SubgraphPartitioner] Liveness node #{} "
                                "introduces CopyInput@{} into the subgraph "
                                "partition for {}",
                                i,
                                index,
                                graph.getGraphString());

      } else if (isCallOpWithStatus(graph, node, OpStatus::CopyOutput)) {

        // A CallOp copying output.
        syncCalls();
        nodes.push_back({op, {CallOpPartType::CopyOutput, 0, index, 0}});

        logging::devicex::trace("[SubgraphPartitioner] Liveness node #{} "
                                "introduces CopyOutput@{} into the subgraph "
                                "partition for {}",
                                i,
                                index,
                                graph.getGraphString());

      } else if (isCallOpWithStatus(graph, node, OpStatus::CopyModified)) {

        // A CallOp copying modified input.
        syncCalls();
        nodes.push_back({op, {CallOpPartType::CopyModified, index, 0, 0}});

        logging::devicex::trace("[SubgraphPartitioner] Liveness node #{} "
                                "introduces CopyModified@{} into the subgraph "
                                "partition for {}",
                                i,
                                index,
                                graph.getGraphString());

      } else if (isCopyCallSubgraphPart(graph, node, activeCallOp)) {

        // We're dealing with an op in a subgraph called by our active CallOp.

        CallOp *callOp = dynamic_cast<CallOp *>(node.getOp());
        if (callOp == nullptr) {

          // The op is a normal op in a child subgraph. Normal ops are currently
          // always lowered over *exactly* 1 subgraph part.

          SubgraphPartIndex begin = getOpSubgraphPartBegin(op);
          SubgraphPartIndex end   = getOpSubgraphPartEnd(op);

          if (end - begin != 1) {
            // We shouldn't need multiple subgraph parts for non-call op.
            throw internal_error("[SubgraphPartitioner] Lowering over multiple "
                                 "subgraph parts is only supported for call "
                                 "ops ({} suggests lowering over {} parts)",
                                 op->debugName(),
                                 end - begin);
          }

          if (begin <= finalisedPart) {
            // Implies we need a call to a subgraph part we already called.
            auto actGraphStr = activeCallOp->getCalledGraph().getGraphString();
            throw internal_error("[SubgraphPartitioner] Invalid schedule for "
                                 "{}. The schedule for {} (op {}) would need "
                                 "to call subgraph part {} of {} more than "
                                 "once with this schedule.",
                                 actGraphStr,
                                 graph.getGraphString(),
                                 activeCallOp->debugName(),
                                 begin,
                                 actGraphStr);
          }

          logging::devicex::trace("[SubgraphPartitioner] Liveness node #{} "
                                  "implies the need for a CallSubgraphPart({}) "
                                  " in the subgraph partition for {}",
                                  i,
                                  end - 1,
                                  graph.getGraphString());

          // Make sure to insert calls to these subgraph parts in the schedule.
          discoveredPart = std::max(discoveredPart, end - 1);

        } else {

          // The op is a CallOp. Multiple liveness nodes are associated with
          // CallOps (Enter/Exit/CopyInputs/CopyOutputs/CopyModified). We need
          // the *exact* subgraph part the node we're looking at is in as we
          // don't want to include calls to subgraph parts we don't need yet.
          // Use the call op schedule to determine the correct part.

          const auto &callOpSchedule = getCallOpSchedule(callOp);
          bool doLog                 = false;

          for (const auto &entry : callOpSchedule) {
            const auto &callOpPart = std::get<0>(entry);
            const auto &part       = std::get<1>(entry);

            if (node.getStatus() == OpStatus::CopyInput &&
                callOpPart.type == CallOpPartType::CopyInput &&
                node.getIndex() == callOpPart.inIndex) {

              // Found relevant CopyInput in call op schedule, use the part.
              discoveredPart = part;
              doLog          = true;

            } else if (node.getStatus() == OpStatus::CopyOutput &&
                       callOpPart.type == CallOpPartType::CopyOutput &&
                       node.getIndex() == callOpPart.outIndex) {

              // Found relevant CopyOutput in call op schedule, use the part.
              discoveredPart = part;
              doLog          = true;

            } else if (node.getStatus() == OpStatus::CopyModified &&
                       callOpPart.type == CallOpPartType::CopyModified &&
                       node.getIndex() == callOpPart.inIndex) {

              // Found relevant CopyModified in call op schedule, use the part.
              discoveredPart = part;
              doLog          = true;
            }
          }

          if (doLog) {
            logging::devicex::trace("[SubgraphPartitioner] Liveness node #{} "
                                    "implies a need for "
                                    "CallSubgraphPart({}) in the subgraph "
                                    "partition for {}",
                                    i,
                                    discoveredPart,
                                    graph.getGraphString());
          }
        }

      } else if (!nodes.empty() && isParentCopy(graph, node, callstackSize)) {

        // Add a boundary here as there's a parent copy. Note that it's possible
        // this happens multiple times without adding a node but that's okay.
        boundaries.insert(nodes.size());

        logging::devicex::trace("[SubgraphPartitioner] Liveness node #{} "
                                "implies the need for partition boundary in "
                                "the subgraph partition for {}",
                                i,
                                graph.getGraphString());

      } else if (isNormal(graph, node)) {

        // It's a normal op.
        nodes.push_back({op, {CallOpPartType::Undefined, 0, 0, 0}});

        logging::devicex::trace("[SubgraphPartitioner] Liveness node #{} "
                                "introduces a normal op node into the subgraph "
                                "partition for {}",
                                i,
                                graph.getGraphString());
      }
    }
  }

  // Log the partition we found.
  logSubgraphPartitionTmp(graph, result);

  return result;
}

SubgraphPartitioner::SubgraphPartitionTmp
SubgraphPartitioner::mergeSubgraphPartitions(
    const Graph &graph,
    const SubgraphPartitionTmp &part0,
    const SubgraphPartitionTmp &part1) {
  // Check the sequence is the same. This has no effect on the result
  // but we do this to defensively check our assumptions.
  const auto &seq0 = std::get<0>(part0);
  const auto &seq1 = std::get<0>(part1);

  if (seq0.size() != seq1.size()) {
    throw internal_error("[SubgraphPartitioner] Subgraph partition for {} "
                         "unexpectedly differs in length from previous "
                         "partition ({} != {})",
                         graph.getGraphString(),
                         seq0.size(),
                         seq1.size());
  }

  for (size_t t = 0; t < seq0.size(); ++t) {
    auto node0 = seq0.at(t);
    auto node1 = seq1.at(t);

    if ((node0.op != node1.op) ||
        (node0.callOpPart.type != node1.callOpPart.type) ||
        (node0.callOpPart.inIndex != node1.callOpPart.inIndex) ||
        (node0.callOpPart.outIndex != node1.callOpPart.outIndex) ||
        (node0.callOpPart.subgraphPartIndex !=
         node1.callOpPart.subgraphPartIndex)) {
      throw internal_error("[SubgraphPartitioner] Subgraph partition for {} "
                           "unexpectedly differs from a previous "
                           "partition (see position {})",
                           graph.getGraphString(),
                           t);
    }
  }

  // Okay, the sequences match. All we have to do now is combine boundaries.
  const auto &boundaries0 = std::get<1>(part0);
  const auto &boundaries1 = std::get<1>(part1);

  SubgraphPartitioner::SubgraphPartitionTmp result;
  std::get<0>(result) = seq0;
  std::get<1>(result).insert(boundaries0.begin(), boundaries0.end());
  std::get<1>(result).insert(boundaries1.begin(), boundaries1.end());

  return result;
}

SubgraphPartitioner::SubgraphPartition
SubgraphPartitioner::finaliseSubgraphPartition(
    const SubgraphPartitionTmp &part) {
  SubgraphPartition result;

  const auto &nodes      = std::get<0>(part);
  const auto &boundaries = std::get<1>(part);

  // Start with no boundaries.
  for (const auto &node : nodes) {
    result.push_back({node, 0});
  }

  for (const size_t &boundary : boundaries) {
    // Apply each boundary in turn.
    std::transform(result.begin() + boundary,
                   result.end(),
                   result.begin() + boundary,
                   [](auto entry) {
                     return std::make_pair(std::get<0>(entry),
                                           std::get<1>(entry) + 1);
                   });
  }

  return result;
}

void SubgraphPartitioner::populateCache(const Graph &graph) {
  if (!liveness)
    throw internal_error("[SubgraphPartitioner] LivenessAnalyzer not set");

  // Use opCache to see if we previously determined this mapping.
  auto it = cache.find(graph.id.str());
  if (it == cache.end()) {
    // Determine and remember graph schedule.
    bool partitionable = isPartitionable(graph);
    auto graphSchedule = determineSubgraphPartition(graph, partitionable);
    cache.insert(it, {graph.id.str(), graphSchedule});
  }
}

void SubgraphPartitioner::populateCacheForCalledGraphs(const Graph &graph) {
  const auto &schedule = liveness->getGraphOpSchedule(graph.id);
  for (auto op : schedule) {
    for (auto graph : op->getCalledGraphs()) {
      populateCache(*graph);
    }
  }
}

std::ostream &operator<<(std::ostream &os,
                         const SubgraphPartitioner::CallOpPartType &type) {
  using CallOpPartType = SubgraphPartitioner::CallOpPartType;
  switch (type) {
  case CallOpPartType::CopyInput:
    os << "CopyInput";
    break;
  case CallOpPartType::CopyOutput:
    os << "CopyOutput";
    break;
  case CallOpPartType::CopyModified:
    os << "CopyModified";
    break;
  case CallOpPartType::CallSubgraphPart:
    os << "CallSubgraphPart";
    break;
  case CallOpPartType::Undefined:
  default:
    os << "Undefined";
    break;
  }
  return os;
}

std::ostream &operator<<(std::ostream &os,
                         const SubgraphPartitioner::CallOpPart &part) {
  using CallOpPartType = SubgraphPartitioner::CallOpPartType;
  os << part.type;
  switch (part.type) {
  case CallOpPartType::CopyInput:
  case CallOpPartType::CopyModified:
    os << "@" << part.inIndex;
    break;
  case CallOpPartType::CopyOutput:
    os << "@" << part.outIndex;
    break;
  case CallOpPartType::CallSubgraphPart:
    os << "(" << part.subgraphPartIndex << ")";
    break;
  case CallOpPartType::Undefined:
  default:
    break;
  }
  return os;
}

std::ostream &operator<<(std::ostream &os,
                         const SubgraphPartitioner::Node &node) {
  using CallOpPartType = SubgraphPartitioner::CallOpPartType;
  switch (node.callOpPart.type) {
  case CallOpPartType::CopyInput:
  case CallOpPartType::CopyOutput:
  case CallOpPartType::CopyModified:
  case CallOpPartType::CallSubgraphPart:
    os << "[" << node.callOpPart << "] ";
    break;
  case CallOpPartType::Undefined:
  default:
    break;
  }
  os << node.op->debugName();
  return os;
}

void SubgraphPartitioner::logSubgraphPartitionTmp(
    const Graph &graph,
    const SubgraphPartitionTmp &partition) const {
  if (logging::devicex::isEnabled(logging::Level::Trace)) {
    logging::devicex::trace("[SubgraphPartitioner] Extracted a subgraph "
                            "partition for one instance of {}:",
                            graph.getGraphString());
    std::stringstream ss;
    size_t i               = 0;
    SubgraphPartIndex part = 0;
    const auto &nodes      = std::get<0>(partition);
    const auto &boundaries = std::get<1>(partition);
    for (const auto &node : nodes) {
      if (boundaries.find(i) != boundaries.end())
        part++;

      // "[SubgraphPartitioner] #43: [CopyInput@1] <some call op> "
      std::stringstream ss;
      ss << "[SubgraphPartitioner] ";
      ss << "#" << i++;
      ss << "->" << part;
      ss << ": " << node;
      logging::devicex::trace(ss.str());
    }
  }
}

void SubgraphPartitioner::logSubgraphPartition(
    const Graph &graph,
    const SubgraphPartition &partition,
    bool partitionable) const {
  if (logging::devicex::isEnabled(logging::Level::Debug)) {

    logging::devicex::debug("[SubgraphPartitioner] Determined subgraph "
                            "partition for {} ({}):",
                            graph.getGraphString(),
                            partitionable ? "partitionable"
                                          : "unpartitionable");
    size_t i = 0;
    for (const auto &tup : partition) {
      // "[SubgraphPartitioner] #43->5: [CopyInput@1] <some call op>"
      std::stringstream ss;
      ss << "[SubgraphPartitioner] ";
      ss << "#" << i++;
      ss << "->" << std::get<1>(tup);
      ss << ": " << std::get<0>(tup);
      logging::devicex::debug(ss.str());
    }
  }
}

} // namespace liveness
} // namespace popart
