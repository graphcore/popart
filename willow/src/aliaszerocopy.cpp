// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <poprithms/logging/timepartitionlogger.hpp>
#include <popart/aliasesmap.hpp>
#include <popart/aliaszerocopy.hpp>
#include <popart/chains.hpp>
#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/maxclique.hpp>
#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/op/call.hpp>
#include <popart/op/init.hpp>
#include <popart/op/ipucopy.hpp>
#include <popart/op/loop.hpp>
#include <popart/pointercomparators.hpp>
#include <popart/region.hpp>
#include <popart/tensors.hpp>
#include <popart/topocons.hpp>

#include <algorithm>
#include <queue>

#include <boost/icl/interval.hpp>
#include <boost/icl/interval_set.hpp>

namespace popart {
namespace liveness {

namespace {

std::pair<int64_t, int64_t> bisect(const std::vector<int64_t> &indices,
                                   const int64_t &index) {
  auto lowerIt = std::lower_bound(indices.begin(), indices.end(), index);
  auto upperIt = std::upper_bound(indices.begin(), indices.end(), index);
  return {std::distance(indices.begin(), lowerIt),
          std::distance(indices.begin(), upperIt)};
}

} // namespace

// Right-open intervals of tensor liveness
typedef boost::icl::interval_set<int64_t> BoostInterval;
class IntervalsImpl : public BoostInterval {};
typedef boost::icl::interval_set<int64_t>::interval_type IntervalImpl;

Intervals::Intervals() : intervals(new IntervalsImpl) {}

Intervals::Intervals(const Intervals &other)
    : intervals(new IntervalsImpl(*other.intervals)) {}

Intervals::~Intervals() {}

bool Intervals::empty() const { return intervals->empty(); }

Intervals Intervals::operator&(const Intervals &other) const {
  Intervals newInterval;
  *static_cast<BoostInterval *>(newInterval.intervals.get()) |=
      *static_cast<BoostInterval *>(intervals.get());
  *static_cast<BoostInterval *>(newInterval.intervals.get()) &=
      *static_cast<BoostInterval *>(other.intervals.get());
  return newInterval;
}

Intervals &Intervals::operator=(const Intervals &other) {
  *static_cast<BoostInterval *>(intervals.get()) =
      *static_cast<BoostInterval *>(other.intervals.get());
  return *this;
}

Intervals &Intervals::operator+=(const Intervals &other) {
  *static_cast<BoostInterval *>(intervals.get()) +=
      *static_cast<BoostInterval *>(other.intervals.get());
  return *this;
}

bool Intervals::operator==(const Intervals &other) const {
  return boost::icl::is_element_equal(
      *static_cast<BoostInterval *>(intervals.get()),
      *static_cast<BoostInterval *>(other.intervals.get()));
}

bool Intervals::operator!=(const Intervals &other) const {
  return !(*this == other);
}

std::ostream &operator<<(std::ostream &os, const Intervals &intervals) {
  os << *static_cast<BoostInterval *>(intervals.intervals.get());
  return os;
}

void Intervals::insert(int64_t s, int64_t e) {
  intervals->insert(IntervalImpl(s, e));
}

AliasZeroCopy::AliasZeroCopy(const Ir *ir_, const LivenessAnalyzer *analyzer_)
    : ir(ir_), analyzer(analyzer_) {

  // Selectively turn off alias zero copy of tensors containing certain strings
  // helps debugging aliasing issues
  // excludeTensorByName.insert("Attention");
  // excludeTensorByName.insert("Dense");
}

void AliasZeroCopy::apply() {

  const auto lifetimeTimer =
      ir->timePartitionLogger().scopedStopwatch("AliasZeroCopy");

  logging::devicex::debug("[AliasZeroCopy] Started.");

  AliasesMap aliasesMap{ir};

  // Record all tensors that are fully aliased in the IR
  for (const Graph *g : ir->getGraphSchedule()) {
    irAliases.addAllAliases(aliasesMap.getAliases(g->id));
  }
  proposedAliases = irAliases;
  activeAliases   = irAliases;

  disabledNodes.resize(analyzer->getOpScheduleSize(), false);

  disableDeadCodeNodes();

  std::map<const Graph *, int64_t> beforeGraphMap;

  std::map<
      std::tuple<int64_t, double, InIndex, OutIndex, GraphId>,
      std::vector<Op *>,
      std::greater<std::tuple<int64_t, double, InIndex, OutIndex, GraphId>>>
      graphIOPriorityMap;

  for (const Graph *graph0 : ir->getGraphSchedule()) {
    for (const Graph *graph1 : graph0->getCalledGraphs()) {
      beforeGraphMap[graph1] += 1;
    }
  }

  std::queue<const Graph *> graphQueue;
  graphQueue.push(&(ir->getMainGraph()));

  {
    int64_t i = ir->getGraphSchedule().size();
    while (!graphQueue.empty()) {
      const Graph *graph0 = graphQueue.front();
      graphQueue.pop();

      logging::devicex::debug("[AliasZeroCopy] Processing I/O of graph {}",
                              graph0->id);

      for (const Graph *graph1 : graph0->getCalledGraphs()) {
        beforeGraphMap[graph1]--;
        if (beforeGraphMap[graph1] == 0) {
          graphQueue.push(graph1);
        }
      }

      auto &callSiteOps = analyzer->getGraphCallSites(graph0->id);
      if (!callSiteOps.empty()) {
        auto num_inputs  = graph0->getInputIds().size();
        auto num_outputs = graph0->getOutputIds().size();
        for (InIndex in = 0; in < num_inputs; ++in) {
          auto input   = graph0->getTensors().get(graph0->getInputId(in));
          double bytes = input->info.nbytes();
          // Prioritize aliasing candidates by graph priority, bytes copied and
          // number of call sites
          graphIOPriorityMap.insert(
              {{i,
                bytes / (1024.0 * 1024.0) * callSiteOps.size(),
                in,
                -1,
                graph0->id},
               callSiteOps});
        }
        for (OutIndex out = 0; out < num_outputs; ++out) {
          auto output  = graph0->getTensors().get(graph0->getOutputId(out));
          double bytes = output->info.nbytes();
          // Prioritize aliasing candidates by graph priority, bytes copied and
          // number of call sites
          graphIOPriorityMap.insert(
              {{i,
                bytes / (1024.0 * 1024.0) * callSiteOps.size(),
                -1,
                out,
                graph0->id},
               callSiteOps});
        }
        // Next graph has lower priority for being processed
        --i;
      }
    }
  }

  // TODO(T51075): Revisit allowing aliasing between InitOp outputs.
  // Init tensor aliasing
  // std::map<int64_t, std::set<Tensor *, PTensorCmp>, std::greater<int64_t>>
  //     initCandidates;
  // for (Op *op : ir->getAllOps()) {
  //   if (dynamic_cast<InitOp *>(op)) {
  //     Tensor *init = op->output->tensor(InitOp::getOutIndex());
  //     initCandidates[init->info.nelms()].insert(init);
  //   }
  // }
  // for (auto &cacheCandidate : initCandidates) {
  //   processTensorAliasGroups(cacheCandidate.second);
  // }

  // Graph input/output aliasing
  for (auto &priorityEntry : graphIOPriorityMap) {
    InIndex sgInIndex   = std::get<2>(priorityEntry.first);
    OutIndex sgOutIndex = std::get<3>(priorityEntry.first);
    const Graph *sgraph = &ir->getGraph(std::get<4>(priorityEntry.first));
    std::vector<Op *> callSiteOps = priorityEntry.second;

    if (sgInIndex >= 0) {
      logging::devicex::trace(
          "[AliasZeroCopy] Subgraph: {}, input index: {}, call sites: {}",
          sgraph->id.str(),
          sgInIndex,
          callSiteOps.size());

      std::set<Tensor *, PTensorCmp> parentGraphTensors;
      for (Op *callSite : callSiteOps) {
        if (SubgraphOp *sgOp = dynamic_cast<SubgraphOp *>(callSite)) {
          InIndex opInIndex = sgOp->subgraphInToOpInIndex(sgInIndex);
          if (sgOp->input->hasIndex(opInIndex)) {
            Tensor *t  = callSite->input->tensor(opInIndex);
            Tensor *at = findAliasableTensor(t);
            if (at) {
              parentGraphTensors.insert(at);
            }
          }
        }
      }
      if (SubgraphOp *sgOp = dynamic_cast<SubgraphOp *>(callSiteOps.front())) {
        InIndex opInIndex = sgOp->subgraphInToOpInIndex(sgInIndex);
        if (sgOp->input->hasIndex(opInIndex)) {
          Tensor *t = sgOp->input->tensor(opInIndex);
          // This is the only possible tensor that can alias from the parent
          // graph to the subgraph; if this is viable can be checked later on.
          std::pair<Tensor *, Tensor *> parentToSubgraphTensors = {
              t, sgraph->getTensors().get(sgraph->getInputId(sgInIndex))};

          processTensorAliasGroups(parentGraphTensors);

          if (checkSubgraphInputCompatible(parentToSubgraphTensors.first,
                                           parentToSubgraphTensors.second)) {
            insertAlias(parentToSubgraphTensors.first,
                        parentToSubgraphTensors.second);
          }
        }
      }
    }

    if (sgOutIndex >= 0) {
      logging::devicex::trace(
          "[AliasZeroCopy] Subgraph: {}, output index: {}, call sites: {}",
          sgraph->id.str(),
          sgOutIndex,
          callSiteOps.size());

      std::set<Tensor *, PTensorCmp> parentGraphTensors;
      for (Op *callSite : callSiteOps) {
        if (SubgraphOp *sgOp = dynamic_cast<SubgraphOp *>(callSite)) {
          OutIndex opOutIndex = sgOp->subgraphOutToOpOutIndex(sgOutIndex);
          if (callSite->output->hasIndex(opOutIndex)) {
            Tensor *at = callSite->output->tensor(opOutIndex);
            if (checkSubgraphOutputCompatible(
                    sgraph->getTensors().get(sgraph->getOutputId(sgOutIndex)),
                    at)) {
              // Parent graph tensors that can be aliased from subgraph tensor
              parentGraphTensors.insert(at);
            }
          }
        }
      }

      // Largest compatible group that can be aliased among each other and with
      // the subgraph tensor
      auto group = processTensorAliasGroups(parentGraphTensors);

      for (Tensor *parent : group) {
        insertAlias(sgraph->getTensors().get(sgraph->getOutputId(sgOutIndex)),
                    parent);
      }
    }
  }

  logPostIRAliases();

  logging::devicex::debug("[AliasZeroCopy] Done.");
}

void AliasZeroCopy::disableDeadCodeNodes() {
  bool changed = true;
  // Disable superfluous nodes, back to front

  auto updateAssociatedIntervals = [this](const LivenessNode &node) {
    std::set<TensorId> used = node.usedTensorIds();
    for (auto &id : used) {
      logging::devicex::trace(
          "[AliasZeroCopy] Updating intervals for {}: {}", node, id);
      Tensor *t = ir->getTensor(id);
      // Force update cache for all associated tensors & their aliases
      getCandidateLivenessIntervals(t, ProducerInterval::Ignore, true);
    }
  };

  auto disableNode =
      [this, &changed, &updateAssociatedIntervals](int64_t index) {
        if (!disabledNodes[index]) {
          logging::devicex::trace("[AliasZeroCopy] Disabling node {}",
                                  analyzer->getOpScheduleAt(index));
          changed              = true;
          disabledNodes[index] = true;
          updateAssociatedIntervals(analyzer->getOpScheduleAt(index));
        }
      };

  int64_t pass = 0;
  while (changed) {
    logging::devicex::debug("[AliasZeroCopy] disableDeadCodeNodes pass {}",
                            pass);
    ++pass;
    changed = false;
    for (int64_t i = analyzer->getOpScheduleSize() - 1; i >= 0; --i) {
      auto &node = analyzer->getOpScheduleAt(i);
      // Probing: Does the tensor need to be live after the producer?
      Intervals probe;
      probe.insert(i, i + 1);

      switch (node.getStatus()) {
      case OpStatus::CopyInput: {
        auto tensorIds = node.getTensorIds();
        Tensor *t      = ir->getTensor(tensorIds.second);
        auto liveness =
            getCandidateLivenessIntervals(t, ProducerInterval::Ignore);

        // If the input tensor is not required to be live after CopyInput,
        // then the CopyInput is not required either
        if (!doOverlap(liveness, probe)) {
          disableNode(i);
        }
        break;
      }
      case OpStatus::CopyLoopCarried: {
        auto tensorIds = node.getTensorIds();
        // First: output of iteration #0
        // Second: input to iteration #1 <- produced
        Tensor *t = ir->getTensor(tensorIds.second);
        auto liveness =
            getCandidateLivenessIntervals(t, ProducerInterval::Ignore);

        // If the input tensor is not required to be live after CopyLoopCarried,
        // then the CopyLoopCarried is not required either
        if (!doOverlap(liveness, probe)) {
          disableNode(i);
        }
        break;
      }
      case OpStatus::CopyModified: {
        auto tensorIds = node.getTensorIds();
        Tensor *t      = ir->getTensor(tensorIds.first);
        auto liveness =
            getCandidateLivenessIntervals(t, ProducerInterval::Ignore);

        // If the modified tensor is not required to be live after CopyModified,
        // then the CopyModified is not required either
        if (!doOverlap(liveness, probe)) {
          disableNode(i);
        }
        break;
      }
      case OpStatus::CopyOutput: {
        auto tensorIds = node.getTensorIds();
        Tensor *t      = ir->getTensor(tensorIds.first);
        auto liveness =
            getCandidateLivenessIntervals(t, ProducerInterval::Ignore);

        // If the input tensor is not required to be live after CopyInput,
        // then the CopyInput is not required either
        if (!doOverlap(liveness, probe)) {
          disableNode(i);
        }
        break;
      }
      case OpStatus::Exit:
      case OpStatus::Normal: {
        if (!node.getOp()->hasSideEffect()) {
          // Side-effect free node

          if (node.getStatus() == OpStatus::Exit) {
            // When using a JustInTime copying strategy an output tensor may not
            // be live by the time you get to the exit node of a call, as
            // outputs may have been copied early. We increase the probe
            // interval for calls here to ensure we do not inadvertently prune
            // out calls we think have no effect.
            int64_t j      = analyzer->getCallSiteLinksInvAt(i).front();
            auto enterNode = analyzer->getOpScheduleAt(j);
            if (enterNode.getStatus() != OpStatus::Enter) {
              throw error("[AliasZeroCopy] findFront: OpStatus {} unexpected.",
                          static_cast<int>(enterNode.getStatus()));
            }
            probe = Intervals();
            probe.insert(j, i + 1);
          }

          bool disable = true;
          for (auto &out : node.getOp()->output->tensorMap()) {
            auto liveness = getCandidateLivenessIntervals(
                out.second, ProducerInterval::Ignore);
            // If none of the outputs are required to be live after this node
            // ...
            disable &= !doOverlap(liveness, probe);
          }
          for (auto &in : node.getOp()->input->tensorMap()) {
            auto liveness = getCandidateLivenessIntervals(
                in.second, ProducerInterval::Ignore);
            // ... and none of the modified inputs either ...
            if (node.getOp()->modifiesIndex(in.first)) {
              disable &= !doOverlap(liveness, probe);
            }
          }
          // ... then the Exit/Normal node is not required either
          if (disable) {
            disableNode(i);
          }
        }
        break;
      }
      case OpStatus::Enter: {
        // Enter coupled to Exit
        int64_t j     = analyzer->getCallSiteLinksAt(i).front();
        auto exitNode = analyzer->getOpScheduleAt(j);
        if (exitNode.getStatus() != OpStatus::Exit) {
          throw error("[AliasZeroCopy] findFront: OpStatus {} unexpected.",
                      static_cast<int>(exitNode.getStatus()));
        }
        if (disabledNodes[j]) {
          // If exit is disabled -> disable enter
          disableNode(i);
        }
        break;
      }
      default:
        break;
      }
    }
  }

  // Check if a node is required across all call sites
  for (int64_t i = 0; i < analyzer->getOpScheduleSize(); ++i) {
    auto &node = analyzer->getOpScheduleAt(i);

    requiredNodes[{node.getOp(), node.getStatus(), node.getIndex()}] |=
        !disabledNodes[i];
  }

  // Update disabledNodes with the final setting of requiredNodes.
  // This ensures future liveness analysis is consistent with the lowered
  // program.
  for (int64_t i = 0; i < analyzer->getOpScheduleSize(); ++i) {
    auto &node = analyzer->getOpScheduleAt(i);

    auto required =
        requiredNodes[{node.getOp(), node.getStatus(), node.getIndex()}];
    disabledNodes[i] = !required;

    logging::opx::trace("[AliasZeroCopy] index: {} node: {} disabled: {}",
                        i,
                        node,
                        static_cast<int>(!required));
  }
}

int64_t AliasZeroCopy::findStart(Tensor *consumedTensor,
                                 int64_t scheduleIndex) const {
  auto &tensorIndices = analyzer->getScheduleIndices(consumedTensor);

  auto i     = bisect(tensorIndices, scheduleIndex).first;
  auto index = tensorIndices.at(
      std::min(i, static_cast<int64_t>(tensorIndices.size() - 1)));

  // Walk back on schedule until the closest previous
  // tensor producer or consumer is found
  // Note that the node/op can exist multiple times in the schedule,
  // if the node/op is located in a subgraph
  auto opScheduleEntry = &analyzer->getOpScheduleAt(index);
  while (!(opScheduleEntry->isProducerOf(consumedTensor) ||
           opScheduleEntry->isConsumerOf(consumedTensor)) ||
         disabledNodes[index] || index == scheduleIndex) {
    --i;
    if (i < 0) {
      return -1;
    }
    index           = tensorIndices.at(i);
    opScheduleEntry = &analyzer->getOpScheduleAt(index);
  }

  return index;
}

Intervals
AliasZeroCopy::getLivenessIntervals(Tensor *t,
                                    ProducerInterval producerInterval) {
  logging::devicex::trace("[AliasZeroCopy] Getting interval of tensor {}",
                          t->id);
  Intervals intervals;
  auto insertInterval = [&](int64_t s, int64_t e) {
    if (s < 0) {
      throw error("[AliasZeroCopy] Interval starts below 0: " +
                  std::to_string(s));
    }
    if (e > analyzer->getOpScheduleSize()) {
      throw error("[AliasZeroCopy] Interval ends above schedule size: " +
                  std::to_string(e) + " > " +
                  std::to_string(analyzer->getOpScheduleSize()));
    }
    if (e < s) {
      throw error("[AliasZeroCopy] Interval ends before start: " +
                  std::to_string(e) + " < " + std::to_string(s));
    }
    intervals.insert(s, e);
  };

  auto &tensorScheduleIndices = analyzer->getScheduleIndices(t);

  if ((!t->hasProducer() && t->getGraph().id == ir->getMainGraph().id) ||
      t->tensorType() == TensorType::Const) {
    // Being conservative and keeping main graph tensors without producer
    // always live
    insertInterval(0, analyzer->getOpScheduleSize());
    return intervals;
  }

  auto anchors = ir->getRootAnchors();
  if (std::find(anchors.begin(), anchors.end(), t->id) != anchors.end()) {
    // Being conservative and keeping anchored tensors always live
    insertInterval(0, analyzer->getOpScheduleSize());
    return intervals;
  }

  auto &mainOutputs = ir->getMainGraph().getOutputIds();
  if (std::find(mainOutputs.begin(), mainOutputs.end(), t->id) !=
      mainOutputs.end()) {
    // Being conservative and keeping main graph output tensors always live
    insertInterval(0, analyzer->getOpScheduleSize());
    return intervals;
  }

  if (t->hasProducer() && !(t->getProducer()->settings.executionContext ==
                                ExecutionContext::Normal ||
                            t->getProducer()->settings.executionContext ==
                                ExecutionContext::Subgraph)) {
    // Being conservative and keeping tensors that are touched outside the
    // training loop always live
    insertInterval(0, analyzer->getOpScheduleSize());
    return intervals;
  }

  for (Op *c : t->consumers.getOps()) {
    if (!(c->settings.executionContext == ExecutionContext::Normal ||
          c->settings.executionContext == ExecutionContext::Subgraph)) {
      // Being conservative and keeping tensors that are touched outside the
      // training loop always live
      insertInterval(0, analyzer->getOpScheduleSize());
      return intervals;
    }
  }

  for (auto scheduleIndex : tensorScheduleIndices) {
    auto opScheduleEntry = &analyzer->getOpScheduleAt(scheduleIndex);

    // Handle producers, graph inputs
    if (producerInterval != ProducerInterval::Ignore &&
        !disabledNodes[scheduleIndex] && opScheduleEntry->isProducerOf(t)) {
      insertInterval(scheduleIndex, scheduleIndex + 1);
    }

    // Handle consumers, graph outputs, loop carried dependencies, copy modified
    if (!disabledNodes[scheduleIndex] && opScheduleEntry->isConsumerOf(t)) {
      int64_t startIndex = findStart(t, scheduleIndex);
      if (startIndex > -1 && !opScheduleEntry->getOp()->overwritesTensor(t)) {
        insertInterval(startIndex, scheduleIndex);
      }
      if (opScheduleEntry->getOp()->input->contains(t)) {
        auto inIndices = opScheduleEntry->getOp()->input->indices(t);
        for (auto inIndex : inIndices) {
          if (producerInterval != ProducerInterval::Ignore &&
              opScheduleEntry->getOp()->modifiesIndex(inIndex)) {
            insertInterval(scheduleIndex, scheduleIndex + 1);
          }
        }
      }
    }
  }

  return intervals;
}

std::set<Tensor *, PTensorCmp>
AliasZeroCopy::getAliasedTensors(const Aliases &aliases,
                                 std::set<Tensor *, PTensorCmp> tensors,
                                 bool fullyAliased) const {
  std::set<Tensor *, PTensorCmp> aliased;
  for (Tensor *t0 : tensors) {
    aliased.insert(t0);

    auto aliasedTensorMap = aliases.aliasChainsFrom(t0);
    for (auto &chain : aliasedTensorMap) {
      Tensor *t1 = chain.first;

      auto fullRegion0 = view::Region::getFull(t0->info.shape());
      auto regions0    = chain.second.apply(fullRegion0);
      auto fullRegion1 = view::Region::getFull(t1->info.shape());
      auto regions1    = aliases.getChainsFromTo(t1, t0).apply(fullRegion1);

      bool accepted = false;

      if (fullyAliased) {
        accepted = true;
        accepted &= t0->info.shape() == t1->info.shape();
        accepted &= view::mergeRegions(regions0).front() == fullRegion1;
        accepted &= view::mergeRegions(regions1).front() == fullRegion0;
      } else {
        accepted =
            std::any_of(regions0.begin(),
                        regions0.end(),
                        [](const view::Region &r) { return !r.isEmpty(); }) ||
            std::any_of(regions1.begin(),
                        regions1.end(),
                        [](const view::Region &r) { return !r.isEmpty(); });
      }

      if (accepted) {
        aliased.insert(t1);
      }
    }
  }
  return aliased;
}

std::set<Tensor *, PTensorCmp>
AliasZeroCopy::getProposedAliasedTensors(std::set<Tensor *, PTensorCmp> tensors,
                                         bool fullyAliased) const {
  return getAliasedTensors(proposedAliases, tensors, fullyAliased);
}

std::set<Tensor *, PTensorCmp>
AliasZeroCopy::getActiveAliasedTensors(std::set<Tensor *, PTensorCmp> tensors,
                                       bool fullyAliased) const {
  return getAliasedTensors(activeAliases, tensors, fullyAliased);
}

void AliasZeroCopy::activateAlias(Tensor *ta, Tensor *tb) {
  logging::devicex::trace(
      "[AliasZeroCopy] Activating post IR alias: {} aliases to {}",
      ta->id,
      tb->id);

  // Full bidirectional alias
  activeAliases.updateAliases(
      ta,
      tb,
      {view::Region::getFull(ta->info.shape())},
      [](const view::Region &r) { return view::Regions(1, r); },
      [](const view::Region &r) { return view::Regions(1, r); });
}

void AliasZeroCopy::insertAlias(Tensor *ta, Tensor *tb) {
  bool excluded = false;
  for (std::string name : excludeTensorByName) {
    excluded |= (ta->id.find(name) != std::string::npos ||
                 tb->id.find(name) != std::string::npos);
  }

  logging::devicex::trace(
      "[AliasZeroCopy] Inserting post IR alias: {} aliases to {} {}",
      ta->id,
      tb->id,
      excluded ? "(excluded)" : "");

  if (!excluded) {
    // Full bidirectional alias
    proposedAliases.updateAliases(
        ta,
        tb,
        {view::Region::getFull(ta->info.shape())},
        [](const view::Region &r) { return view::Regions(1, r); },
        [](const view::Region &r) { return view::Regions(1, r); });
    postIRAliases[ta].insert(tb);
    postIRAliases[tb].insert(ta);
  } else {
    printLivenessIntervals({ta, tb}, ProducerInterval::Enforce);
  }
}

void AliasZeroCopy::removePostIRAliases(Tensor *t0) {
  auto it0 = postIRAliases.find(t0);
  if (it0 != postIRAliases.end()) {
    for (auto t1 : it0->second) {
      auto it1 = postIRAliases.find(t1);
      if (it1 != postIRAliases.end() &&
          it1->second.find(t0) != it1->second.end()) {
        it1->second.erase(t0);
      }
    }
    postIRAliases.erase(t0);
  }
}

std::set<Tensor *, PTensorCmp>
AliasZeroCopy::getPostIRAliases(Tensor *t) const {
  std::set<Tensor *, PTensorCmp> aliases;
  auto it = postIRAliases.find(t);
  if (it != postIRAliases.end()) {
    aliases = it->second;
  }
  return aliases;
}

std::set<Tensor *, PTensorCmp>
AliasZeroCopy::getTensorsWithPostIRAliases() const {
  std::set<Tensor *, PTensorCmp> aliases;
  for (auto &it : postIRAliases) {
    aliases.insert(it.first);
  }
  return aliases;
}

void AliasZeroCopy::logPostIRAliases() {
  // Report post-IR aliased tensors
  if (logging::shouldLog(logging::Module::devicex, logging::Level::Debug)) {
    for (Tensor *t0 : getTensorsWithPostIRAliases()) {
      auto tensors = getPostIRAliases(t0);
      if (tensors.size() > 1) {
        std::vector<TensorId> ids;
        ids.reserve(tensors.size());
        for (Tensor *t1 : tensors) {
          ids.push_back(t1->id);
        }
        logging::devicex::debug(
            "[AliasZeroCopy] PostIRAliases: {} aliases to [{}]",
            t0->id,
            logging::join(ids.begin(), ids.end(), ", "));
      }
    }
  }
}

Intervals
AliasZeroCopy::getCandidateLivenessIntervals(Tensor *startTensor,
                                             ProducerInterval producerInterval,
                                             bool forceUpdateCache) {
  std::set<Tensor *, PTensorCmp> aliasedTensors;
  aliasedTensors.insert(startTensor);

  aliasedTensors = getProposedAliasedTensors(aliasedTensors, false);

  Intervals combinedIntervals;

  for (Tensor *t : aliasedTensors) {
    auto it = candidateLivenessIntervalsMap.find({t, producerInterval});
    if (it == candidateLivenessIntervalsMap.end() || forceUpdateCache) {
      Intervals candidateIntervals = getLivenessIntervals(t, producerInterval);
      candidateLivenessIntervalsMap[{t, producerInterval}] = candidateIntervals;
      combinedIntervals += candidateIntervals;
    } else {
      combinedIntervals += it->second;
    }
  }

  return combinedIntervals;
}

bool AliasZeroCopy::checkCandidatesCompatible(Tensor *ta, Tensor *tb) {
  bool compatible =
      (ta->info == tb->info && ta->getVirtualGraphIdAndTileSetUnsafe() ==
                                   tb->getVirtualGraphIdAndTileSetUnsafe());
  if (!compatible)
    return false;

  auto tensors = getProposedAliasedTensors({ta}, true);
  if (tensors.find(tb) != tensors.end()) {
    return true;
  }

  bool overlapping = AliasZeroCopy::doOverlap(
      getCandidateLivenessIntervals(ta), getCandidateLivenessIntervals(tb));
  return !overlapping;
}

bool AliasZeroCopy::checkSubgraphInputCompatible(Tensor *ta, Tensor *tb) {
  bool compatible =
      (ta->info == tb->info && ta->getVirtualGraphIdAndTileSetUnsafe() ==
                                   tb->getVirtualGraphIdAndTileSetUnsafe());
  if (!compatible) {
    return false;
  }

  bool overlapping = AliasZeroCopy::doOverlap(
      getCandidateLivenessIntervals(ta), getCandidateLivenessIntervals(tb));

  for (Op *c : ta->consumers.getOps()) {
    auto indices = c->input->indices(ta);
    for (const Graph *sgraph : c->getCalledGraphs()) {
      if (sgraph->id == tb->getGraph().id) {
        auto &callSiteOps = analyzer->getGraphCallSites(sgraph->id);
        logging::devicex::trace(
            "[AliasZeroCopy] Subgraph: {} ({} -> {}) with {} call sites.",
            sgraph->id,
            ta->id,
            tb->id,
            callSiteOps.size());

        bool conflict = false;
        bool modified = false;

        for (InIndex opInIndex : indices) {
          InIndex sgInIndex = opInIndex;

          if (SubgraphOp *sgOp =
                  dynamic_cast<SubgraphOp *>(callSiteOps.front())) {
            sgInIndex = sgOp->opInToSubgraphInIndex(opInIndex);
          }

          if (sgraph->getInputId(sgInIndex) == tb->id) {
            for (Op *callSiteOp : callSiteOps) {
              Tensor *tc   = callSiteOp->input->tensor(opInIndex);
              auto aliased = getProposedAliasedTensors({ta}, true);
              if (tc->id != ta->id && aliased.find(tc) == aliased.end() &&
                  (AliasZeroCopy::doOverlap(
                      getCandidateLivenessIntervals(ta),
                      getCandidateLivenessIntervals(tc)))) {
                // If both ta and tc are used as inputs to SubgraphOp,
                // but ta and tc overlap in liveness
                logging::devicex::trace("[AliasZeroCopy] Conflict: {} {} -> {}",
                                        ta->id,
                                        tc->id,
                                        tb->id);
                conflict = true;
              }
            }

            if (overlapping) {
              // If tb is modified in the subgraph,
              // and the parent graph and subgraph tensors are live
              // at the same time, they can't be aliased.
              auto aliases = getProposedAliasedTensors({tb}, false);
              aliases.insert(tb);

              // If any region of tb or it's aliases is modified
              modified |=
                  std::any_of(aliases.begin(), aliases.end(), [](Tensor *t) {
                    return t->isModified();
                  });
            }
          }
        }
        logging::devicex::trace("[AliasZeroCopy] Overlapping: {} {}, "
                                "conflict: {}, overlapping: {}, modified: {}",
                                ta->id,
                                tb->id,
                                conflict,
                                overlapping,
                                modified);
        return !conflict && (!overlapping || !modified);
      }
    }
  }
  return false;
}

bool AliasZeroCopy::checkSubgraphOutputCompatible(Tensor *ta, Tensor *tb) {
  bool compatible =
      (ta->info == tb->info && ta->getVirtualGraphIdAndTileSetUnsafe() ==
                                   tb->getVirtualGraphIdAndTileSetUnsafe());
  if (!compatible) {
    return false;
  }

  // If ta overlaps with tb
  bool overlapping = AliasZeroCopy::doOverlap(
      getCandidateLivenessIntervals(ta), getCandidateLivenessIntervals(tb));

  if (!tb->hasProducer() || !dynamic_cast<SubgraphOp *>(tb->getProducer())) {
    return false;
  }

  Op *producer = tb->getProducer();
  auto index   = producer->output->indices(tb).front();

  for (const Graph *sgraph : producer->getCalledGraphs()) {
    if (sgraph->getOutputId(index) == ta->id) {
      auto aliased = getProposedAliasedTensors({ta}, true);
      if (overlapping && aliased.find(tb) == aliased.end()) {
        // Overlapping and not already aliased
        return false;
      } else {
        // Not overlapping or already aliased
        return true;
      }
    }
  }
  return false;
}

std::vector<Tensor *> AliasZeroCopy::processTensorAliasGroups(
    std::set<Tensor *, PTensorCmp> proposedTensor) {

  if (logging::shouldLog(logging::Module::devicex, logging::Level::Trace)) {
    std::vector<TensorId> ids;
    ids.reserve(proposedTensor.size());
    for (Tensor *proposed : proposedTensor) {
      ids.push_back(proposed->id);
    }
    logging::devicex::trace("Processing alias group: {}", ids);
  }

  if (proposedTensor.empty())
    return std::vector<Tensor *>{};

  std::vector<Tensor *> acceptedTensors;

  auto preFiltered = proposedTensor.size();

  // Filter out already aliased - reduces cost of clique algorithm
  for (Tensor *t0 : proposedTensor) {
    auto tensors        = getProposedAliasedTensors({t0}, true);
    bool alreadyAliased = false;
    for (Tensor *t1 : acceptedTensors) {
      if (tensors.find(t1) != tensors.end()) {
        alreadyAliased = true;
        break;
      }
    }
    if (!alreadyAliased) {
      acceptedTensors.push_back(t0);
    }
  }

  auto postFiltered = acceptedTensors.size();

  logging::devicex::debug(
      "[AliasZeroCopy] Group candidates: before: {}, after: {}, "
      "elements per tensor: {}",
      preFiltered,
      postFiltered,
      (*acceptedTensors.begin())->info.nelms());

  graphclique::AGraph ag(static_cast<int>(acceptedTensors.size()));

  for (int i = 0; i < acceptedTensors.size(); ++i) {
    for (int j = 0; j < i; ++j) {
      if (checkCandidatesCompatible(acceptedTensors[i], acceptedTensors[j])) {
        ag.addEdge(i, j);
      }
    }
  }

  graphclique::MaxClique mq(ag);
  auto mcliques = mq.getMaximumCliques(1, ag.numVertices());

  std::vector<std::vector<Tensor *>> cliques;
  for (auto mclique : mcliques) {
    std::vector<Tensor *> clique(mclique.size());
    for (size_t i = 0; i < mclique.size(); ++i) {
      clique[i] = acceptedTensors[mclique[i]];
      for (size_t j = 0; j < i; ++j) {
        if (!ag.getEdge(mclique[i], mclique[j])) {
          throw error("[AliasZeroCopy] Clique contains invalid result.");
        }
        insertAlias(acceptedTensors[mclique[i]], acceptedTensors[mclique[j]]);
      }
    }
    logging::devicex::trace("[AliasZeroCopy] Clique size: {}", clique.size());
    cliques.push_back(clique);
  }

  if (logging::shouldLog(logging::Module::devicex, logging::Level::Trace)) {
    for (auto &clique : cliques) {
      std::vector<TensorId> ids;
      ids.reserve(clique.size());
      for (Tensor *t : clique) {
        ids.push_back(t->id);
      }
      logging::devicex::trace("[AliasZeroCopy] Clique: {}", ids);
    }
  }

  logging::devicex::debug("[AliasZeroCopy] Maximum clique size: {}",
                          cliques.front().size());
  // Return the largest maximum clique
  return cliques.front();
}

Tensor *AliasZeroCopy::findAliasableTensor(Tensor *t) {
  while (t->hasProducer() && !dynamic_cast<SubgraphOp *>(t->getProducer()) &&
         !dynamic_cast<InitOp *>(t->getProducer())) {
    Op *op            = t->getProducer();
    bool found_next   = false;
    OutIndex outIndex = op->output->indices(t).front();
    for (auto &in : op->input->indicesMap()) {
      for (InIndex inIndex : in.second) {
        auto aliases = op->aliases(inIndex, outIndex);
        if (!aliases.empty() && aliases.front().nelms() == t->info.nelms() &&
            op->outInfo(outIndex).shape() == op->inInfo(inIndex).shape()) {
          // Fully aliased
          found_next = true;
          t          = op->input->tensor(inIndex);
        }
        if (found_next)
          break;
      }
      if (found_next)
        break;
    }
    if (!found_next) {
      // Chain ends without aliasable tensor
      return static_cast<Tensor *>(nullptr);
    }
  }

  // t is aliasable
  return t;
}

void AliasZeroCopy::printLivenessIntervals(
    std::set<Tensor *, PTensorCmp> tensors,
    ProducerInterval producerInterval) {
  if (logging::shouldLog(logging::Module::devicex, logging::Level::Trace)) {
    std::stringstream ss;

    ss << std::endl;

    for (size_t i = 0; i < analyzer->getOpScheduleSize(); ++i) {
      ss << std::min(9UL, analyzer->getOpScheduleAt(i).getCallStack().size());
    }

    ss << std::endl;

    for (Tensor *t0 : tensors) {
      auto livenessIntervals =
          getCandidateLivenessIntervals(t0, producerInterval);
      size_t j = 0;
      for (IntervalsImpl::iterator it = livenessIntervals.intervals->begin();
           it != livenessIntervals.intervals->end();
           it++) {
        if (it->lower() < 0) {
          throw error("[AliasZeroCopy] Interval starts below 0.");
        }
        if (it->upper() > analyzer->getOpScheduleSize()) {
          throw error("[AliasZeroCopy] Interval ends above schedule size.");
        }

        while (j < it->lower()) {
          ss << "_";
          ++j;
        }
        while (j < it->upper()) {
          ss << "*";
          ++j;
        }
      }
      while (j < analyzer->getOpScheduleSize()) {
        ss << "_";
        ++j;
      }
      ss << " (" << t0->id << ")";
      ss << std::endl;
    }
    logging::devicex::trace("[AliasZeroCopy] Liveness intervals {}", ss.str());
  }
}

bool AliasZeroCopy::doOverlap(const Intervals &aIntervals,
                              const Intervals &bIntervals) {
  return !((aIntervals & bIntervals).empty());
}

bool AliasZeroCopy::nodeRequired(Op *op, OpStatus status, int index) const {
  auto it = requiredNodes.find({op, status, index});
  return it == requiredNodes.end() || it->second;
}

bool AliasZeroCopy::opRequired(Op *op) const {
  auto ite = requiredNodes.find({op, OpStatus::Enter, 0});
  auto itn = requiredNodes.find({op, OpStatus::Normal, 0});

  return (ite == requiredNodes.end() && itn == requiredNodes.end()) ||
         (ite != requiredNodes.end() && ite->second) ||
         (itn != requiredNodes.end() && itn->second);
}

bool AliasZeroCopy::copyInputRequired(Op *op, InIndex inIndex) const {
  return nodeRequired(op, OpStatus::CopyInput, inIndex);
}

bool AliasZeroCopy::copyLoopCarriedRequired(Op *op, OutIndex outIndex) const {
  return nodeRequired(op, OpStatus::CopyLoopCarried, outIndex);
}

bool AliasZeroCopy::copyModifiedRequired(Op *op, InIndex inIndex) const {
  return nodeRequired(op, OpStatus::CopyModified, inIndex);
}

bool AliasZeroCopy::copyOutputRequired(Op *op, OutIndex outIndex) const {
  return nodeRequired(op, OpStatus::CopyOutput, outIndex);
}

} // namespace liveness
} // namespace popart
