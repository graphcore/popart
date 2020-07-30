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
#include <popart/op/remote.hpp>
#include <popart/region.hpp>
#include <popart/tensors.hpp>
#include <popart/topocons.hpp>

#include <queue>

#include <boost/icl/interval.hpp>
#include <boost/icl/interval_set.hpp>

namespace popart {
namespace liveness {

// Closed intervals of tensor liveness
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

void Intervals::operator+=(const Intervals &other) {
  *static_cast<BoostInterval *>(intervals.get()) +=
      *static_cast<BoostInterval *>(other.intervals.get());
}

void Intervals::insert(int64_t s, int64_t e) {
  intervals->insert(IntervalImpl(s, e));
}

AliasZeroCopy::AliasZeroCopy(const Ir *ir_, const LivenessAnalyzer *analyzer_)
    : ir(ir_), analyzer(analyzer_) {

  // Selectively turn off alias zero copy of tensors containing certain strings
  // helps debugging aliasing issues
  // excludeTensorByName.insert("Z/Reshape:0/1__t6__re");
}

void AliasZeroCopy::apply() {
  logging::devicex::debug("[AliasZeroCopy] Started.");

  // Record all tensors that are fully aliased in the IR
  for (const Graph *g : ir->getGraphSchedule()) {
    irAliases.addAllAliases(g->getTensors().getAliases());
  }
  proposedAliases = irAliases;
  activeAliases   = irAliases;

  std::map<const Graph *, int64_t> beforeGraphMap;

  std::map<std::tuple<int64_t, double, const Graph *, InIndex, OutIndex>,
           std::vector<Op *>,
           std::greater<
               std::tuple<int64_t, double, const Graph *, InIndex, OutIndex>>>
      graphIOPriorityMap;

  for (const Graph *graph0 : ir->getGraphSchedule()) {
    for (const Graph *graph1 : graph0->getCalledGraphs()) {
      beforeGraphMap[graph1] += 1;
    }
  }

  std::queue<const Graph *> graphQueue;
  graphQueue.push(&(ir->getMainGraph()));

  {
    int64_t i = 0;
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
      auto num_inputs   = graph0->getInputIds().size();
      auto num_outputs  = graph0->getOutputIds().size();
      for (InIndex in = 0; in < num_inputs; ++in) {
        auto input   = graph0->getTensors().get(graph0->getInputId(in));
        double bytes = input->info.nbytes();
        // Prioritize aliasing candidates by graph priority, bytes copied and
        // number of call sites
        graphIOPriorityMap.insert(
            {{i,
              bytes / (1024.0 * 1024.0) * callSiteOps.size(),
              graph0,
              in,
              -1},
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
              graph0,
              -1,
              out},
             callSiteOps});
      }
      // Next graph has lower priority for being processed
      i--;
    }
  }

  // Cache tensor aliasing
  std::map<int64_t, std::set<Tensor *, PTensorCmp>, std::greater<int64_t>>
      cacheCandidates;

  for (Op *op : ir->getAllOps()) {
    if (dynamic_cast<InitOp *>(op)) {
      Tensor *cache = op->output->tensor(InitOp::getOutIndex());
      if (cache->getTensorTypeInfo()->type() == TensorType::Cache) {
        cacheCandidates[cache->info.nelms()].insert(cache);
      }
    }
  }

  for (auto &cacheCandidate : cacheCandidates) {
    processTensorAliasGroups(cacheCandidate.second);
  }

  // Graph input/output aliasing
  for (auto &priorityEntry : graphIOPriorityMap) {
    const Graph *sgraph           = std::get<2>(priorityEntry.first);
    InIndex inIndex               = std::get<3>(priorityEntry.first);
    OutIndex outIndex             = std::get<4>(priorityEntry.first);
    std::vector<Op *> callSiteOps = priorityEntry.second;

    if (inIndex >= 0) {
      logging::devicex::trace(
          "[AliasZeroCopy] Subgraph: {}, input index: {}, call sites: {}",
          sgraph->id.str(),
          inIndex,
          callSiteOps.size());

      std::set<Tensor *, PTensorCmp> parentGraphTensors;
      for (Op *callSite : callSiteOps) {
        Tensor *t  = callSite->input->tensor(inIndex);
        Tensor *at = findAliasableTensor(t);
        if (at) {
          parentGraphTensors.insert(at);
        }
      }
      // This is the only possible tensor that can alias from the parent graph
      // to the subgraph; if this is viable can be checked later on.
      std::pair<Tensor *, Tensor *> parentToSubgraphTensors = {
          callSiteOps.front()->input->tensor(inIndex),
          sgraph->getTensors().get(sgraph->getInputId(inIndex))};

      // printLivenessIntervals(parentGraphTensors);

      processTensorAliasGroups(parentGraphTensors);

      if (checkSubgraphInputCompatible(parentToSubgraphTensors.first,
                                       parentToSubgraphTensors.second)) {
        insertAlias(parentToSubgraphTensors.first,
                    parentToSubgraphTensors.second);
      }
    }

    if (outIndex >= 0) {
      logging::devicex::trace(
          "[AliasZeroCopy] Subgraph: {}, output index: {}, call sites: {}",
          sgraph->id.str(),
          outIndex,
          callSiteOps.size());

      std::set<Tensor *, PTensorCmp> parentGraphTensors;
      for (Op *callSite : callSiteOps) {
        Tensor *at = callSite->output->tensor(outIndex);
        if (checkSubgraphOutputCompatible(
                sgraph->getTensors().get(sgraph->getOutputId(outIndex)), at)) {
          // Parent graph tensors that can be aliased from subgraph tensor
          parentGraphTensors.insert(at);
        }
      }

      // Largest compatible group that can be aliased among each other and with
      // the subgraph tensor
      auto group = processTensorAliasGroups(parentGraphTensors);

      // Aliasing to one of the group's tensor aliases to all of them
      if (group.size() > 0) {
        insertAlias(sgraph->getTensors().get(sgraph->getOutputId(outIndex)),
                    (*group.begin()));
      }
    }
  }

  generatePostIRAliases();

  logging::devicex::debug("[AliasZeroCopy] Done.");
}

int64_t AliasZeroCopy::findStart(Tensor *consumedTensor,
                                 int64_t scheduleIndex) const {

  if (consumedTensor->hasProducer()) {
    // Producer: Find producer OP on schedule
    auto index = scheduleIndex;
    // Walk back on schedule until the closest previous
    // tensor producer is found
    // Note that the producer can exist multiple times in the schedule,
    // if the producer is located in a subgraph
    auto opScheduleEntry = &analyzer->getOpScheduleAt(index);
    do {
      --index;
      if (index < 0)
        throw error("[AliasZeroCopy] Schedule index below 0 (producer)");
      opScheduleEntry = &analyzer->getOpScheduleAt(index);
    } while (std::get<0>(*opScheduleEntry).back() !=
                 consumedTensor->getProducer() ||
             (std::get<1>(*opScheduleEntry) != OpStatus::Exit &&
              std::get<1>(*opScheduleEntry) != OpStatus::Normal));
    // Verify this is a normal or exit entry
    Op *producer = std::get<0>(analyzer->getOpScheduleAt(index)).back();
    auto status  = std::get<1>(analyzer->getOpScheduleAt(index));
    switch (status) {
    case OpStatus::CopyInput:
    case OpStatus::CopyOutput:
    case OpStatus::CopyModified:
    case OpStatus::Enter: {
      throw error("[AliasZeroCopy] findStart: OpStatus {} unexpected.",
                  static_cast<int>(status));
      // break;
    }
    case OpStatus::Exit: {
      OutIndex outIndex = producer->output->indices(consumedTensor).front();
      do {
        --index;
      } while (std::get<2>(analyzer->getOpScheduleAt(index)) != outIndex);
      break;
    }
    case OpStatus::Normal: {
      break;
    }
    }

    return index;
  } else {
    if (consumedTensor->getGraph().id == ir->getMainGraph().id) {
      // No producer: main graph input tensor
      // Main graph start in schedule
      return 0L;
    } else {
      InIndex inIndex = 0;
      // No producer: non-main graph input tensor
      auto &subgraphInputIds = consumedTensor->getGraph().getInputIds();
      for (InIndex i = 0; i < subgraphInputIds.size(); ++i) {
        if (subgraphInputIds[i] == consumedTensor->id) {
          inIndex = i;
        }
      }

      auto &callStack = std::get<0>(analyzer->getOpScheduleAt(scheduleIndex));
      auto callStackSize   = callStack.size();
      int64_t index        = scheduleIndex;
      auto opScheduleEntry = &analyzer->getOpScheduleAt(index);
      do {
        // Walk back on schedule until call site
        // (caller of the current subgraph) is found
        // this can be an Exit or Enter depending on which subgraph of an
        // op this is
        --index;
        if (index < 0)
          throw error("[AliasZeroCopy] Schedule index below 0 (no producer)");
        opScheduleEntry = &analyzer->getOpScheduleAt(index);
      } while (std::get<0>(*opScheduleEntry).size() >= callStackSize ||
               std::get<1>(*opScheduleEntry) != OpStatus::CopyInput ||
               std::get<2>(*opScheduleEntry) != inIndex);
      return index;
    }
  }
}

int64_t AliasZeroCopy::findFront(Tensor *consumedTensor,
                                 int64_t scheduleIndex,
                                 bool modified) const {

  auto opScheduleEntry = &analyzer->getOpScheduleAt(scheduleIndex);
  Op *consumer         = std::get<0>(*opScheduleEntry).back();
  OpStatus status      = std::get<1>(*opScheduleEntry);
  auto inIndices       = consumer->input->indices(consumedTensor);

  if (status != OpStatus::Normal && status != OpStatus::Enter) {
    throw error("[AliasZeroCopy] findFront: OpStatus {} unexpected.",
                static_cast<int>(status));
  }

  if (status == OpStatus::Normal) {
    // Normal Op
    return scheduleIndex;
  } else {
    // Subgraphing Op
    if (modified) {
      // Get last entry for subgraph (OpStatus::Exit)
      auto index = analyzer->getCallSiteLinksAt(scheduleIndex).back();
      // Verify end is an exit entry
      if (std::get<1>(analyzer->getOpScheduleAt(index)) != OpStatus::Exit) {
        throw error("[AliasZeroCopy] OpStatus::Exit expected.");
      }
      // Walk back to find OpStatus::CopyModified for that input
      do {
        --index;
        if (index < 0)
          throw error("[AliasZeroCopy] Schedule index below 0 (no producer)");
        opScheduleEntry = &analyzer->getOpScheduleAt(index);
      } while (std::get<1>(*opScheduleEntry) != OpStatus::CopyModified ||
               std::get<2>(*opScheduleEntry) != inIndices.front());
      return index;
    } else {
      // Walk forward to find OpStatus::CopyInput for that input
      auto index = scheduleIndex;
      do {
        ++index;
        if (index >= analyzer->getOpScheduleSize())
          throw error("[AliasZeroCopy] Schedule index above schedule size ({})",
                      analyzer->getOpScheduleSize());
        opScheduleEntry = &analyzer->getOpScheduleAt(index);
      } while (std::get<1>(*opScheduleEntry) != OpStatus::CopyInput ||
               std::get<2>(*opScheduleEntry) != inIndices.front());
      return index;
    }
  }
}

Intervals AliasZeroCopy::getLivenessIntervals(Tensor *t) {
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

  std::map<int64_t, std::pair<Op *, std::vector<InIndex>>> consumers;

  if ((!t->hasProducer() && t->getGraph().id == ir->getMainGraph().id) ||
      t->getTensorTypeInfo()->type() == TensorType::Const) {
    // Being conservative and keeping main graph tensors without producer
    // always live
    insertInterval(0, analyzer->getOpScheduleSize());
    return intervals;
  }

  for (Op *c : t->consumers.getOps()) {
    for (auto &scheduleIndex : analyzer->getScheduleIndices(c)) {
      consumers.insert({scheduleIndex, {c, c->input->indices(t)}});
    }
  }

  // Handle graph outputs
  auto graphOutputIds = t->getGraph().getOutputIds();
  for (OutIndex outIndex = 0; outIndex < graphOutputIds.size(); ++outIndex) {
    TensorId outId = graphOutputIds[outIndex];
    if (t->id == outId) {
      auto &callSiteOps = analyzer->getGraphCallSites(t->getGraph().id);
      for (Op *callSiteOp : callSiteOps) {
        auto &scheduleIndices = analyzer->getScheduleIndices(callSiteOp);
        for (int64_t scheduleIndex : scheduleIndices) {
          // TODO: This code is over-conservative for multiple subgraphs
          // (e.g. if operation).

          // Last exit entry
          auto callSiteLinks   = analyzer->getCallSiteLinksAt(scheduleIndex);
          int64_t index        = callSiteLinks.back();
          auto opScheduleEntry = &analyzer->getOpScheduleAt(index);

          // Walk backwards to find OpStatus::CopyOutput for that output
          do {
            --index;
            if (index < 0)
              throw error(
                  "[AliasZeroCopy] Schedule index below 0 (no copy output)");
            opScheduleEntry = &analyzer->getOpScheduleAt(index);
          } while (std::get<1>(*opScheduleEntry) != OpStatus::CopyOutput ||
                   std::get<2>(*opScheduleEntry) != outIndex);

          // Find producer index
          int64_t startIndex = findStart(t, index);
          insertInterval(startIndex, index);
        }
      }
    }
  }

  int64_t last_overwrite = 0;
  // Iterate over consumers in order
  for (auto consumer : consumers) {
    // Establish the consumed tensor
    Tensor *consumedTensor =
        consumer.second.first->inTensor(consumer.second.second.front());

    auto indices = consumer.second.second;
    Op *op       = consumer.second.first;

    // Overwrite: The consumer will overwrite the tensor without reading it
    bool overwrite = false;

    for (auto index : indices) {
      auto modifies = op->modifies(index);

      if (modifies.size() > 0 && !modifies.front().isEmpty()) {
        overwrite = true;
      }

      if ((modifies.size() == 0) ||
          (std::any_of(
              modifies.begin(), modifies.end(), [](const view::Region &r) {
                return r.getAccessType() == view::AccessType::Read ||
                       r.getAccessType() == view::AccessType::ReadWrite ||
                       r.isEmpty();
              }))) {
        // Consumer without modification, or modifications after reading,
        // we assume the tensor is read and must therefore be live
        overwrite = false;
      }
    }

    int64_t front = findFront(consumedTensor, consumer.first, false);
    int64_t start = std::max(last_overwrite, findStart(consumedTensor, front));

    if (overwrite) {
      last_overwrite = findFront(consumedTensor, consumer.first, true);
    } else {
      insertInterval(start, front);
    }
  }

  return intervals;
}

std::set<Tensor *, PTensorCmp>
AliasZeroCopy::getAliasedTensors(Aliases &aliases,
                                 std::set<Tensor *, PTensorCmp> tensors,
                                 bool fullyAliased) {
  std::set<Tensor *, PTensorCmp> aliased;
  for (Tensor *t : tensors) {
    aliased.insert(t);

    auto aliasedTensorMap = aliases.aliasChainsFrom(t);
    for (auto &chain : aliasedTensorMap) {
      auto fullRegion = view::Region::getFull(t->info.shape());
      auto regions    = chain.second.apply(fullRegion);

      bool accepted = false;

      if (fullyAliased) {
        accepted = true;
        accepted &= t->info.shape() == chain.first->info.shape();
        accepted &= view::mergeRegions(regions).front() ==
                    view::Region::getFull(chain.first->info.shape());
      } else {
        accepted = std::any_of(regions.begin(),
                               regions.end(),
                               [](view::Region &r) { return !r.isEmpty(); });
      }

      if (accepted) {
        aliased.insert(chain.first);
      }
    }
  }
  return aliased;
}

std::set<Tensor *, PTensorCmp>
AliasZeroCopy::getProposedAliasedTensors(std::set<Tensor *, PTensorCmp> tensors,
                                         bool fullyAliased) {
  return getAliasedTensors(proposedAliases, tensors, fullyAliased);
}

std::set<Tensor *, PTensorCmp>
AliasZeroCopy::getActiveAliasedTensors(std::set<Tensor *, PTensorCmp> tensors,
                                       bool fullyAliased) {
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
  } else {
    printLivenessIntervals({ta, tb});
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

void AliasZeroCopy::generatePostIRAliases() {
  for (Tensor *tensor : proposedAliases.getTensors()) {
    auto aliases = getProposedAliasedTensors({tensor}, true);
    postIRAliases[tensor].insert(aliases.begin(), aliases.end());
  }

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

Intervals AliasZeroCopy::getCandidateLivenessIntervals(Tensor *startTensor) {
  std::set<Tensor *, PTensorCmp> aliasedTensors;
  aliasedTensors.insert(startTensor);

  aliasedTensors = getProposedAliasedTensors(aliasedTensors, false);

  Intervals combinedIntervals;

  for (Tensor *t : aliasedTensors) {
    auto it = candidateLivenessIntervalsMap.find(t);
    if (it == candidateLivenessIntervalsMap.end()) {
      Intervals candidateIntervals = getLivenessIntervals(t);
      candidateLivenessIntervalsMap.insert({t, candidateIntervals});
      combinedIntervals += candidateIntervals;
    } else {
      combinedIntervals += it->second;
    }
  }

  return combinedIntervals;
}

bool AliasZeroCopy::checkCandidatesCompatible(Tensor *ta, Tensor *tb) {
  bool compatible =
      (ta->info.shape() == tb->info.shape() &&
       ta->info.dataType() == tb->info.dataType() &&
       ta->getVirtualGraphIdUnsafe() == tb->getVirtualGraphIdUnsafe());
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
      (ta->info.shape() == tb->info.shape() &&
       ta->info.dataType() == tb->info.dataType() &&
       ta->getVirtualGraphIdUnsafe() == tb->getVirtualGraphIdUnsafe());
  if (!compatible)
    return false;

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

        bool conflict            = false;
        bool modified            = false;
        bool fullyCallerModified = false;

        for (auto index : indices) {
          if (sgraph->getInputId(index) == tb->id) {
            for (Op *callSiteOp : callSiteOps) {
              Tensor *tc   = callSiteOp->input->tensor(index);
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
              // If tb is modified or aliased in the subgraph,
              // and the parent graph and subgraph tensors are live
              // at the same time, they can't be aliased.

              // Sufficient to only check IR aliased,
              // since post IR aliasing won't have touched tb yet at this point
              // (due to graphs being processed top-level to bottom-level)
              auto aliases = getProposedAliasedTensors({tb}, false);
              aliases.insert(tb);

              // If any region of tb or it's aliases is modified
              modified = std::any_of(aliases.begin(),
                                     aliases.end(),
                                     [](Tensor *t) { return t->isModified(); });

              auto callerModifiedRegions = c->modifies(index);
              fullyCallerModified |=
                  callerModifiedRegions.size() > 0 &&
                  (callerModifiedRegions.front().getAccessType() ==
                       view::AccessType::ReadWrite ||
                   callerModifiedRegions.front().getAccessType() ==
                       view::AccessType::Write) &&
                  callerModifiedRegions.front().nelms() == ta->info.nelms();
            }
          }
        }
        logging::devicex::trace("[AliasZeroCopy] Overlapping: {} {}, "
                                "modified: {}, caller modified: {}",
                                ta->id,
                                tb->id,
                                modified,
                                fullyCallerModified);
        return !conflict && (!modified || fullyCallerModified);
      }
    }
  }
  return false;
}

bool AliasZeroCopy::checkSubgraphOutputCompatible(Tensor *ta, Tensor *tb) {
  bool compatible =
      (ta->info.shape() == tb->info.shape() &&
       ta->info.dataType() == tb->info.dataType() &&
       ta->getVirtualGraphIdUnsafe() == tb->getVirtualGraphIdUnsafe());
  if (!compatible)
    return false;

  // If ta overlaps with tb
  bool overlapping = AliasZeroCopy::doOverlap(
      getCandidateLivenessIntervals(ta), getCandidateLivenessIntervals(tb));

  if (!tb->hasProducer() || !dynamic_cast<SubgraphOp *>(tb->getProducer()))
    return false;

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
    std::set<Tensor *, PTensorCmp> tensors) {
  if (logging::shouldLog(logging::Module::devicex, logging::Level::Trace)) {
    std::stringstream ss;

    ss << std::endl;

    for (size_t i = 0; i < analyzer->getOpScheduleSize(); ++i) {
      ss << std::min(9UL, std::get<0>(analyzer->getOpScheduleAt(i)).size());
    }

    ss << std::endl;

    for (Tensor *t0 : tensors) {
      auto livenessIntervals = getCandidateLivenessIntervals(t0);
      size_t j               = 0;
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

} // namespace liveness
} // namespace popart
