#include <algorithm>
#include <queue>

#include <popart/graph.hpp>
#include <popart/op.hpp>
#include <popart/scheduler.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>
#include <popart/tensors.hpp>
#include <popart/topocons.hpp>

namespace popart {

namespace {

using OpIdPair = std::pair<Op *, size_t>;

struct OpIdPairCmp {
  bool operator()(OpIdPair const &a, OpIdPair const &b) const {
    return a.first->id < b.first->id;
  }
};

class ComparerCache {
public:
  std::vector<int64_t> memDiffVec;
  std::vector<std::pair<int64_t, int64_t>> opTieIndexVec;
  std::vector<std::string> ioNamesVec;
};

class OpPriorityComparer {
public:
  OpPriorityComparer(const Graph &graph,
                     std::map<Op *, size_t, POpCmp> opIdMap) {
    c = std::make_shared<ComparerCache>();

    c->ioNamesVec.resize(opIdMap.size());
    c->memDiffVec.resize(opIdMap.size());
    c->opTieIndexVec.resize(opIdMap.size());

    for (OpIdPair op_id : opIdMap) {
      c->ioNamesVec[op_id.second]    = ioNames(op_id.first);
      c->memDiffVec[op_id.second]    = memDiff(op_id.first);
      c->opTieIndexVec[op_id.second] = {-1, -1};
    }
    std::vector<OpIdPair> starts;
    std::map<OpIdPair,
             std::pair<std::set<OpIdPair, OpIdPairCmp>,
                       std::set<OpIdPair, OpIdPairCmp>>,
             OpIdPairCmp>
        tops;
    for (auto &op_id : opIdMap) {
      std::set<OpIdPair, OpIdPairCmp> tiedBefores;
      for (auto before : graph.topoCons->getTiedBefores(op_id.first)) {
        tiedBefores.insert({before, opIdMap.at(before)});
      }
      std::set<OpIdPair, OpIdPairCmp> tiedAfters;
      for (auto after : graph.topoCons->getTiedAfters(op_id.first)) {
        tiedAfters.insert({after, opIdMap.at(after)});
      }
      if (tiedBefores.size() == 0 && tiedAfters.size() > 0) {
        starts.push_back(op_id);
      }
      tops[op_id] = {tiedBefores, tiedAfters};
    }

    for (int64_t i = 0; i < starts.size(); ++i) {
      c->opTieIndexVec[starts[i].second] = {i + 1, 0};
    }

    // Run Kahn's algorithm to topologically sort directed subgraphs of
    // tied operations
    uint64_t op_index = 0;
    while (starts.size() > 0) {
      auto &op_id = starts.back();
      starts.pop_back();
      c->opTieIndexVec[op_id.second].second = op_index++;
      for (OpIdPair top : tops[op_id].second) {
        tops[top].first.erase(op_id);
        if (c->opTieIndexVec[top.second] ==
            std::pair<int64_t, int64_t>({-1, -1})) {
          c->opTieIndexVec[top.second] = {c->opTieIndexVec[op_id.second].first,
                                          0};
        } else {
          uint64_t l_group_index =
              std::min(c->opTieIndexVec[op_id.second].first,
                       c->opTieIndexVec[top.second].first);
          uint64_t u_group_index =
              std::max(c->opTieIndexVec[op_id.second].first,
                       c->opTieIndexVec[top.second].first);
          for (auto &opAndIndex : c->opTieIndexVec) {
            if (opAndIndex.first == u_group_index) {
              opAndIndex.first = l_group_index;
            }
          }
          c->opTieIndexVec[top.second] = {l_group_index, 0};
        }
        if (tops[top].first.size() == 0) {
          starts.push_back(top);
        }
      }
    }

    // Operations belonging to the same connected directed subgraph
    std::map<uint64_t, std::set<OpIdPair, OpIdPairCmp>> groups;
    for (auto &op_id : opIdMap) {
      if (c->opTieIndexVec[op_id.second] !=
          std::pair<int64_t, int64_t>({-1, -1})) {
        groups[c->opTieIndexVec[op_id.second].first].insert(op_id);
      }
    }

    for (auto group : groups) {
      auto memdiff = groupMemDiff(opIdMap, group.second);
      for (OpIdPair op_id : group.second) {
        c->memDiffVec[op_id.second] = memdiff;
      }
    }
  }

  // Calculate the memdiff of a single op
  uint64_t memDiff(Op *const &op) const {
    uint64_t diff = 0;
    for (auto &t : op->input->tensors()) {
      diff += t->info.nbytes();
    }

    for (auto &t : op->output->tensors()) {
      diff -= t->info.nbytes();
    }
    return diff;
  }

  // Calculate the memdiff of a group of ops.
  // Only tensors going in/out of the group are counted.
  int64_t groupMemDiff(std::map<Op *, size_t, POpCmp> opIdMap,
                       std::set<OpIdPair, OpIdPairCmp> opIdPairs) const {
    int64_t diff = 0;
    for (OpIdPair op_id : opIdPairs) {
      for (auto &t : op_id.first->input->tensors()) {
        // Tensor has no producer or producer is not internal to the group
        if (!t->hasProducer() ||
            opIdMap.find(t->getProducer()) == opIdMap.end()) {
          diff += t->info.nbytes();
        }
      }
    }

    for (OpIdPair op_id : opIdPairs) {
      for (auto &t : op_id.first->output->tensors()) {
        for (Op *consumer : t->consumers.getOps()) {
          // Tensor has at least one consumer external to the group
          if (opIdMap.find(consumer) == opIdMap.end()) {
            diff -= t->info.nbytes();
            break;
          }
        }
      }
    }
    return diff;
  }

  std::string ioNames(Op *const &op) const {
    std::stringstream ss;
    auto inputMap  = op->input->tensorIdMap();
    auto outputMap = op->output->tensorIdMap();
    std::vector<TensorId> tensorIds;
    tensorIds.reserve(inputMap.size() + outputMap.size());

    for (auto elem : inputMap) {
      tensorIds.push_back(elem.second);
    }

    for (auto elem : outputMap) {
      tensorIds.push_back(elem.second);
    }

    ss << logging::format(
        "{}", logging::join(tensorIds.begin(), tensorIds.end(), "_"));
    return ss.str();
  }

  bool operator()(OpIdPair const &op1, OpIdPair const &op2) const {
    // lexicographic comparison by,
    // 0) ping-pong phase
    // 1) priority
    // 2) n_inputs - n_ouputs
    // 3) op group index and index of the op within the group
    // 4) type (string)
    // 5) input/output names
    // 6) () unique id.
    //
    // Motivation for (2) above is minimization of tensor liveness
    // Motivation for (3) is to keep tied operations together
    // Motivation for (5) is to simplify outlining similarly
    // named/organized structures in different parts of the graph, such as
    // repeating transformer layers.

    auto op1_phase = op1.first->getOptionalPingPongPhase();
    auto op2_phase = op2.first->getOptionalPingPongPhase();

    auto op1_phase_or = op1_phase ? *op1_phase : -1;
    auto op2_phase_or = op2_phase ? *op2_phase : -1;

    auto op1_batchserial = op1.first->getOptionalBatchSerializedPhase();
    auto op2_batchserial = op2.first->getOptionalBatchSerializedPhase();

    auto op1_batchserial_or = op1_batchserial ? *op1_batchserial : -1;
    auto op2_batchserial_or = op2_batchserial ? *op2_batchserial : -1;

    return std::tuple<int,
                      double,
                      int,
                      int,
                      std::pair<uint64_t, uint64_t>,
                      std::string,
                      std::string,
                      OpId>(op2_phase_or,
                            op1.first->priority,
                            op2_batchserial_or,
                            c->memDiffVec[op1.second],
                            c->opTieIndexVec[op2.second],
                            op1.first->opid.type,
                            c->ioNamesVec[op2.second],
                            op1.first->id) <
           std::tuple<int,
                      double,
                      int,
                      int,
                      std::pair<uint64_t, uint64_t>,
                      std::string,
                      std::string,
                      OpId>(op1_phase_or,
                            op2.first->priority,
                            op1_batchserial_or,
                            c->memDiffVec[op2.second],
                            c->opTieIndexVec[op1.second],
                            op2.first->opid.type,
                            c->ioNamesVec[op1.second],
                            op2.first->id);
  }

private:
  std::shared_ptr<ComparerCache> c;
};

class OpContainerBase {
public:
  virtual void pop()          = 0;
  virtual OpIdPair top()      = 0;
  virtual void push(OpIdPair) = 0;
  virtual bool empty()        = 0;
  virtual size_t size()       = 0;
};

class OpContainer : public OpContainerBase {
public:
  void pop() final { impl.pop(); }

  OpIdPair top() final { return impl.front(); }

  void push(OpIdPair opIdPair) final { impl.push(opIdPair); }

  bool empty() final { return impl.empty(); }

  size_t size() final { return impl.size(); }

private:
  std::queue<OpIdPair> impl;
};

class PriorityOpContainer : public OpContainerBase {
public:
  PriorityOpContainer(OpPriorityComparer comparer) : impl(comparer) {}

  void pop() final { impl.pop(); }

  OpIdPair top() final { return impl.top(); }

  void push(OpIdPair opIdPair) final { impl.push(opIdPair); }

  bool empty() final { return impl.empty(); }

  size_t size() final { return impl.size(); }

private:
  std::priority_queue<OpIdPair, std::vector<OpIdPair>, OpPriorityComparer> impl;
};
} // namespace

std::vector<Op *>
Scheduler::getPartialOpSchedule(const OpsBeforeKey &gCons,
                                const Graph &graph,
                                bool respectPriorities,
                                bool respectPingPongPhases) const {

  // note that if gCons has constraints
  // of the form A -> A, the sorting is not complete

  auto &ops     = graph.getOps();
  auto &tensors = graph.getTensors();

  // Ops not enrolled for processing yet
  std::map<Op *, size_t, POpCmp> opIdMap;
  size_t id = 0;
  for (auto &op : ops) {
    opIdMap.emplace(op.second.get(), id);
    ++id;
  }

  auto unprocessedOps = opIdMap;

  // the topological sorting (to construct in this function)
  std::vector<Op *> sorted;
  sorted.reserve(ops.size());
  // ops which have all their input tensors
  // created, and are not waiting for any ops
  // to run before them
  std::shared_ptr<OpContainerBase> opsToProcess;

  if (respectPriorities) {
    logging::ir::trace("[Scheduler] Setting up OP priority comparer.");
    // Initialize comparer class with precomputed values for the given ops
    OpPriorityComparer comparerInstance(graph, opIdMap);
    opsToProcess = std::make_shared<PriorityOpContainer>(comparerInstance);
  } else {
    opsToProcess = std::make_shared<OpContainer>();
  }

  // Number of ops in each PingPong phase
  std::map<PingPongPhase, int> nPhaseOps;
  // map from each op to the number of tensor input
  // indices it is waiting on
  std::vector<int> nIndicesAwaiting(ops.size(), 0);
  // initialise nIndicesAwatings as total
  // number of input indices
  for (auto &op_id : opIdMap) {
    Op *op                         = op_id.first;
    nIndicesAwaiting[op_id.second] = op->input->n();
    if (respectPingPongPhases && op->getOptionalPingPongPhase()) {
      auto phase = op->getOptionalPingPongPhase().get();
      ++nPhaseOps[phase];
      logging::ir::trace("[scheduler] Op {}, phase {}", op->debugName(), phase);
    }
  }

  // the next two variables are needed because of the
  // external constraints: consumer topological constraints
  // and the global constraints passed into the function.
  // (1) map for each op to the number of ops which must
  //     still be inserted before it can it can be inserted
  std::vector<int> nBeforeOpId(ops.size(), 0);
  // (2) map from each op to a list of
  //     ops which are waiting for it
  std::vector<std::vector<OpIdPair>> opsAfterOpId(ops.size());

  // check if the relationship before->after is registered in opsAfterOpId
  auto registered = [&opsAfterOpId](OpIdPair before_id, OpIdPair after_id) {
    return std::find(opsAfterOpId[before_id.second].begin(),
                     opsAfterOpId[before_id.second].end(),
                     after_id) != opsAfterOpId[before_id.second].end();
  };

  for (auto &id_op : ops) {

    // we are going through all the ops, and registering all the
    // constraints which have the op as "after"
    Op *after     = id_op.second.get();
    auto id_after = unprocessedOps[after];

    // we first check the existing constraints on "after"
    for (Op *before : graph.topoCons->getBefores(after)) {
      if (before == after) {
        throw error("[scheduler] Op {} cannot appear before itself",
                    before->debugName());
      }
      auto id_before = unprocessedOps[before];
      if (!registered({before, id_before}, {after, id_after})) {
        logging::ir::trace("[scheduler] Op {} topologically before {}",
                           before->debugName(),
                           after->debugName());
        opsAfterOpId[id_before].push_back({after, id_after});
        ++nBeforeOpId[id_after];
      }
    }

    // and then check any additional global constraints from the user
    auto found = gCons.find(after);
    if (found != gCons.end()) {
      for (auto before : found->second) {
        auto id_before = unprocessedOps[before];
        if (!registered({before, id_before}, {after, id_after})) {
          logging::ir::trace(
              "[scheduler] {} topologically before {} (user defined)",
              before->debugName(),
              after->debugName());
          opsAfterOpId[unprocessedOps[before]].push_back({after, id_after});
          ++nBeforeOpId[id_after];
        }
      }
    }
  }

  auto readyToProcess =
      [&nPhaseOps, &nIndicesAwaiting, &nBeforeOpId](OpIdPair id_op) {
        if (id_op.first->getOptionalPingPongPhase()) {
          auto phase = id_op.first->getOptionalPingPongPhase().get();
          for (int i = 0; i < phase; ++i) {
            if (nPhaseOps[i] > 0)
              return false;
          }
        }

        return (nIndicesAwaiting[id_op.second] == 0 &&
                nBeforeOpId[id_op.second] == 0);
      };

  // processing a tensor involves
  // reducing the counts in `awaiting' for
  // ops which use it, and detecting which
  // ops have nothing left to wait for as a
  // result of such updating.
  auto processTensor = [&opsToProcess,
                        &unprocessedOps,
                        &nIndicesAwaiting,
                        &readyToProcess](Tensor *tensor) {
    for (auto &op_count : tensor->consumers.getMap()) {
      Op *op  = op_count.first;
      auto it = unprocessedOps.find(op);
      if (it != unprocessedOps.end()) {
        nIndicesAwaiting[it->second] -= op_count.second;
        if (readyToProcess({op, it->second})) {
          opsToProcess->push({op, it->second});
          unprocessedOps.erase(op);
        }
      }
    }
  };

  // we will start by processing
  // the tensors which have no producers
  auto t0 = tensors.getNoProducerIds();
  for (auto &id : t0) {
    processTensor(tensors.get(id));
  }
  // also process the graph inputs
  for (auto &id : graph.getInputIds()) {
    processTensor(tensors.get(id));
  }

  // All ops that do not consume anything
  std::vector<OpIdPair> erase;
  for (OpIdPair op_id : unprocessedOps) {
    if (readyToProcess(op_id)) {
      opsToProcess->push(op_id);
      erase.push_back(op_id);
    }
  }
  for (auto id_op : erase)
    unprocessedOps.erase(id_op.first);
  erase.clear();

  while (true) {

    if (unprocessedOps.size() % 100 == 0) {
      logging::trace("Unprocessed: {}, queued: {}",
                     unprocessedOps.size(),
                     opsToProcess->size());
    }

    if (opsToProcess->empty()) {
      break;
    }

    auto op_id_process = opsToProcess->top();
    // logging::ir::trace(
    //    "[scheduler] Scheduling {}, VGID: {}, PingPong phase: {}",
    //    op->debugName(),
    //    op->hasVirtualGraphId() ? op->getVirtualGraphId() : -1,
    //    op->getOptionalPingPongPhase() ?
    //           op->getOptionalPingPongPhase().get() : -1);

    opsToProcess->pop();
    sorted.push_back(op_id_process.first);
    if (op_id_process.first->getOptionalPingPongPhase()) {
      auto phase = op_id_process.first->getOptionalPingPongPhase().get();
      --nPhaseOps[phase];
      if (nPhaseOps[phase] == 0) {
        for (OpIdPair op_id : unprocessedOps) {
          if (readyToProcess(op_id)) {
            opsToProcess->push(op_id);
            erase.push_back(op_id);
          }
        }
        for (auto erase_op_id : erase)
          unprocessedOps.erase(erase_op_id.first);
        erase.clear();
      }
    }

    for (OpIdPair waitingOp : opsAfterOpId[op_id_process.second]) {
      --nBeforeOpId[waitingOp.second];
      if (readyToProcess(waitingOp)) {
        opsToProcess->push(waitingOp);
        unprocessedOps.erase(waitingOp.first);
      }
    }

    for (auto &tensor_indices : op_id_process.first->output->indicesMap()) {
      processTensor(tensor_indices.first);
    }
  }

  if (sorted.size() != graph.getOps().size() &&
      logging::shouldLog(logging::Module::ir, logging::Level::Debug)) {
    for (auto &op : graph.getOps()) {
      auto id = opIdMap[op.second.get()];

      if (std::find(sorted.begin(), sorted.end(), op.second.get()) ==
          sorted.end()) {

        auto optPhase       = op.second.get()->getOptionalPingPongPhase();
        auto phase          = optPhase ? optPhase.get() : -1;
        int nPhaseOpsBefore = 0;
        for (int i = 0; i < phase; ++i)
          nPhaseOpsBefore += nPhaseOps[i];

        std::set<std::string> opDebugNames;
        for (OpIdPair afterOp : opsAfterOpId[id]) {
          opDebugNames.insert(afterOp.first->debugName());
        }

        logging::ir::trace(
            "[scheduler] Failed to schedule {}, "
            "nIndicesAwaiting: {}, nBeforeOpId: {}, opsAfterOpId: {}, "
            "nPhaseOpsBefore: {}",
            op.second->debugName(),
            nIndicesAwaiting[id],
            nBeforeOpId[id],
            opDebugNames,
            nPhaseOpsBefore);
      }
    }
  }

  return sorted;
}

} // namespace popart
