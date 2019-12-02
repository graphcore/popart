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

class OpPriorityComparer {
public:
  OpPriorityComparer(const Graph &graph, std::set<Op *, POpCmp> ops) {
    for (Op *op : ops) {
      ioNamesMap[op] = ioNames(op);
      memDiffMap[op] = memDiff(op);
    }

    std::vector<Op *> starts;
    std::map<Op *,
             std::pair<std::set<Op *, POpCmp>, std::set<Op *, POpCmp>>,
             POpCmp>
        tops;
    for (Op *op : ops) {
      std::set<Op *, POpCmp> tiedBefores;
      for (auto before : graph.topoCons->getTiedBefores(op)) {
        tiedBefores.insert(before);
      }
      std::set<Op *, POpCmp> tiedAfters;
      for (auto after : graph.topoCons->getTiedAfters(op)) {
        tiedAfters.insert(after);
      }
      if (tiedBefores.size() == 0 && tiedAfters.size() > 0) {
        starts.push_back(op);
      }
      tops[op] = {tiedBefores, tiedAfters};
    }

    for (int64_t i = 0; i < starts.size(); ++i) {
      opTieIndexMap[starts[i]] = {i + 1, 0};
    }

    // Run Kahn's algorithm to topologically sort directed subgraphs of
    // tied operations
    uint64_t op_index = 0;
    while (starts.size() > 0) {
      Op *op = starts.back();
      starts.pop_back();
      opTieIndexMap[op].second = op_index++;
      for (Op *top : tops[op].second) {
        tops[top].first.erase(op);
        if (opTieIndexMap.find(top) == opTieIndexMap.end()) {
          opTieIndexMap[top] = {opTieIndexMap[op].first, 0};
        } else {
          uint64_t l_group_index =
              std::min(opTieIndexMap[op].first, opTieIndexMap[top].first);
          uint64_t u_group_index =
              std::max(opTieIndexMap[op].first, opTieIndexMap[top].first);
          for (auto &opAndIndex : opTieIndexMap) {
            if (opAndIndex.second.first == u_group_index) {
              opAndIndex.second.first = l_group_index;
            }
          }
          opTieIndexMap[top] = {l_group_index, 0};
        }
        if (tops[top].first.size() == 0) {
          starts.push_back(top);
        }
      }
    }

    // Operations belonging to the same connected directed subgraph
    std::map<uint64_t, std::set<Op *, POpCmp>> groups;
    for (Op *op : ops) {
      auto entry = opTieIndexMap.find(op);
      if (entry != opTieIndexMap.end()) {
        groups[entry->second.first].insert(entry->first);
      } else {
        // Group 0 for all remaining operations
        opTieIndexMap[op] = {0, 0};
      }
    }

    for (auto group : groups) {
      auto memdiff = groupMemDiff(group.second);
      for (Op *op : group.second) {
        memDiffMap[op] = memdiff;
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
  int64_t groupMemDiff(std::set<Op *, POpCmp> ops) const {
    int64_t diff = 0;
    for (Op *op : ops) {
      for (auto &t : op->input->tensors()) {
        // Tensor has no producer or producer is not internal to the group
        if (!t->hasProducer() || ops.find(t->getProducer()) == ops.end()) {
          diff += t->info.nbytes();
        }
      }
    }

    for (Op *op : ops) {
      for (auto &t : op->output->tensors()) {
        for (Op *consumer : t->consumers.getOps()) {
          // Tensor has at least one consumer external to the group
          if (ops.find(consumer) == ops.end()) {
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

  bool operator()(Op *const &op1, Op *const &op2) const {

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

    auto op1_phase = op1->getOptionalPingPongPhase();
    auto op2_phase = op2->getOptionalPingPongPhase();

    auto op1_phase_or = op1_phase ? *op1_phase : -1;
    auto op2_phase_or = op2_phase ? *op2_phase : -1;

    auto op1_batchserial = op1->getOptionalBatchSerializedPhase();
    auto op2_batchserial = op2->getOptionalBatchSerializedPhase();

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
                            op1->priority,
                            op2_batchserial_or,
                            memDiffMap.at(op1),
                            opTieIndexMap.at(op2),
                            op1->opid.type,
                            ioNamesMap.at(op2),
                            op1->id) < std::tuple<int,
                                                  double,
                                                  int,
                                                  int,
                                                  std::pair<uint64_t, uint64_t>,
                                                  std::string,
                                                  std::string,
                                                  OpId>(op1_phase_or,
                                                        op2->priority,
                                                        op1_batchserial_or,
                                                        memDiffMap.at(op2),
                                                        opTieIndexMap.at(op1),
                                                        op2->opid.type,
                                                        ioNamesMap.at(op1),
                                                        op2->id);
  }

  std::map<Op *, int64_t, POpCmp> memDiffMap;
  std::map<Op *, std::pair<uint64_t, uint64_t>, POpCmp> opTieIndexMap;
  std::map<Op *, std::string, POpCmp> ioNamesMap;
};

std::vector<Op *>
Scheduler::getPartialOpSchedule(const OpsBeforeKey &gCons,
                                const Graph &graph,
                                bool respectPingPongPhases) const {

  // note that if gCons has constraints
  // of the form A -> A, the sorting is not complete

  auto &ops     = graph.getOps();
  auto &tensors = graph.getTensors();

  // Ops not enrolled for processing yet yet
  std::set<Op *, POpCmp> unprocessedOps;
  for (auto &id_op : ops) {
    unprocessedOps.emplace(id_op.second.get());
  }

  // Initialize comparer class with precomputed values for the given ops
  OpPriorityComparer comparerInstance(graph, unprocessedOps);

  // the topological sorting (to construct in this function)
  std::vector<Op *> sorted;
  // ops which have all their input tensors
  // created, and are not waiting for any ops
  // to run before them
  std::priority_queue<Op *, std::vector<Op *>, OpPriorityComparer> opsToProcess(
      comparerInstance);
  // Number of ops in each PingPong phase
  std::map<PingPongPhase, int> nPhaseOps;
  // map from each op to the number of tensor input
  // indices it is waiting on
  std::map<Op *, int, POpCmp> nIndicesAwaiting;
  // initialise nIndicesAwatings as total
  // number of input indices
  for (auto &id_op : ops) {
    Op *op               = id_op.second.get();
    nIndicesAwaiting[op] = op->input->n();
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
  std::map<Op *, int, POpCmp> nBeforeKey;
  // (2) map from each op to a list of
  //     ops which are waiting for it
  std::map<Op *, std::vector<Op *>, POpCmp> opsAfterKey;
  // initialise (1) and (2)
  for (auto &id_op : ops) {
    Op *op          = id_op.second.get();
    nBeforeKey[op]  = 0;
    opsAfterKey[op] = {};
  }

  // check if the relationship before->after is registered in opsAfterKey
  auto registered = [&opsAfterKey](Op *before, Op *after) {
    return std::find(opsAfterKey[before].begin(),
                     opsAfterKey[before].end(),
                     after) != opsAfterKey[before].end();
  };

  for (auto &id_op : ops) {

    // we are going through all the ops, and registering all the
    // constraints which have the op as "after"
    Op *after = id_op.second.get();

    // we first check the existing constraints on "after"
    for (Op *before : graph.topoCons->getBefores(after)) {
      if (before == after) {
        throw error("[scheduler] Op {} cannot appear before itself",
                    before->debugName());
      }
      if (!registered(before, after)) {
        logging::ir::trace("[scheduler] Op {} topologically before {}",
                           before->debugName(),
                           after->debugName());
        opsAfterKey[before].push_back(after);
        ++nBeforeKey[after];
      }
    }

    // and then check any additional global constraints from the user
    auto found = gCons.find(after);
    if (found != gCons.end()) {
      for (auto before : found->second) {
        if (!registered(before, after)) {
          logging::ir::trace(
              "[scheduler] {} topologically before {} (user defined)",
              before->debugName(),
              after->debugName());
          opsAfterKey[before].push_back(after);
          ++nBeforeKey[after];
        }
      }
    }
  }

  auto readyToProcess = [&nPhaseOps, &nIndicesAwaiting, &nBeforeKey](Op *op) {
    if (op->getOptionalPingPongPhase()) {
      auto phase = op->getOptionalPingPongPhase().get();
      for (int i = 0; i < phase; ++i) {
        if (nPhaseOps[i] > 0)
          return false;
      }
    }

    return (nIndicesAwaiting.at(op) == 0 && nBeforeKey.at(op) == 0);
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
      Op *op = op_count.first;
      nIndicesAwaiting[op] -= op_count.second;
      if (readyToProcess(op) &&
          unprocessedOps.find(op) != unprocessedOps.end()) {
        opsToProcess.push(op);
        unprocessedOps.erase(op);
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

  while (true) {
    // All ops that do not consume anything
    std::vector<Op *> erase;
    for (Op *op : unprocessedOps) {
      if (readyToProcess(op)) {
        opsToProcess.push(op);
        erase.push_back(op);
      }
    }
    for (auto op : erase)
      unprocessedOps.erase(op);

    if (opsToProcess.empty()) {
      break;
    }

    auto op = opsToProcess.top();
    // logging::ir::trace(
    //    "[scheduler] Scheduling {}, VGID: {}, PingPong phase: {}",
    //    op->debugName(),
    //    op->hasVirtualGraphId() ? op->getVirtualGraphId() : -1,
    //    op->getOptionalPingPongPhase() ?
    //           op->getOptionalPingPongPhase().get() : -1);

    opsToProcess.pop();
    sorted.push_back(op);
    if (op->getOptionalPingPongPhase()) {
      auto phase = op->getOptionalPingPongPhase().get();
      --nPhaseOps[phase];
    }

    for (Op *waitingOp : opsAfterKey[op]) {
      --nBeforeKey[waitingOp];
      if (readyToProcess(waitingOp)) {
        opsToProcess.push(waitingOp);
        unprocessedOps.erase(waitingOp);
      }
    }

    for (auto &tensor_indices : op->output->indicesMap()) {
      processTensor(tensor_indices.first);
    }
  }

  if (sorted.size() != graph.getOps().size() &&
      logging::shouldLog(logging::Module::ir, logging::Level::Debug)) {
    for (auto &op : graph.getOps()) {
      if (std::find(sorted.begin(), sorted.end(), op.second.get()) ==
          sorted.end()) {

        auto optPhase       = op.second.get()->getOptionalPingPongPhase();
        auto phase          = optPhase ? optPhase.get() : -1;
        int nPhaseOpsBefore = 0;
        for (int i = 0; i < phase; ++i)
          nPhaseOpsBefore += nPhaseOps[i];

        std::set<std::string> opDebugNames;
        for (Op *afterOp : opsAfterKey.at(op.second.get())) {
          opDebugNames.insert(afterOp->debugName());
        }

        logging::ir::debug(
            "[scheduler] Failed to schedule {}, "
            "nIndicesAwaiting: {}, nBeforeKey: {}, opsAfterKey: {}, "
            "nPhaseOpsBefore: {}",
            op.second->debugName(),
            nIndicesAwaiting[op.second.get()],
            nBeforeKey.at(op.second.get()),
            opDebugNames,
            nPhaseOpsBefore);
      }
    }
  }

  return sorted;
}

} // namespace popart
