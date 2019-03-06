#include <algorithm>
#include <queue>

#include <poponnx/ir.hpp>
#include <poponnx/op.hpp>
#include <poponnx/scheduler.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/tensors.hpp>
#include <poponnx/topocons.hpp>

namespace poponnx {

// A note on non-determinism. For maps with
// pointers as keys, iterating through them
// is non-deterministic with the default comparitor.
// To prevent non-determinism in getTopologicallSorted,
// we could use the following non-default comparitor
// everywhere where there is a map with Op pointers,
// and a similar one with Tensor pointers. A fair amount
// of work...
struct POpCmp {
  bool operator()(Op *const &a, Op *const &b) const { return a->id < b->id; }
};

class OpPriorityComparer {
public:
  // total memory of inputs "minus" total memory of outputs
  // TODO T8279 : Op scheduling would be (I think significantly) faster
  // if this value was not computed every time compared.
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

  bool operator()(Op *const &op1, Op *const &op2) const {

    // lexicographic comparison by,
    // 1) priority
    // 2) n_inputs - n_ouputs
    // 3) type (string)
    // 4) unique id.
    //
    // Motivation for (2) above is minimization of tensor liveness

    return std::tuple<double, int, std::string, OpId>(
               op1->priority, memDiff(op1), op1->opid.type, op1->id) <
           std::tuple<double, int, std::string, OpId>(
               op2->priority, memDiff(op2), op2->opid.type, op2->id);
  }
};

std::vector<Op *>
Scheduler::getPartialOpSchedule(const OpsBeforeKey &gCons) const {

  // note that if gCons has constraints
  // of the form A -> A, the sorting is not complete

  auto &ops     = pir->getOps();
  auto &tensors = pir->getTensors();

  // the topological sorting (to construct in this function)
  std::vector<Op *> sorted;
  // ops which have all their input tensors
  // created, and are not waiting for any ops
  // to run before them
  std::priority_queue<Op *, std::vector<Op *>, OpPriorityComparer> opsToProcess;
  // map from each op to the number of tensor input
  // indices it is waiting on
  std::map<Op *, int> nIndicesAwaiting;
  // initialise nIndicesAwatings as total
  // number of input indices
  for (auto &id_op : ops) {
    Op *op               = id_op.second.get();
    nIndicesAwaiting[op] = op->input->n();
  }

  // the next two variables are needed because of the
  // external constraints: consumer topological constraints
  // and the global constraints passed into the function.
  // (1) map for each op to the number of ops which must
  //     still be inserted before it can it can be inserted
  std::map<Op *, int> nBeforeKey;
  // (2) map from each op to a list of
  //     ops which are waiting for it
  std::map<Op *, std::vector<Op *>> opsAfterKey;
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
    for (Op *before : pir->topoCons->getBefores(after)) {
      if (!registered(before, after)) {
        opsAfterKey[before].push_back(after);
        ++nBeforeKey[after];
      }
    }

    // and then check any additional global constraints from the user
    auto found = gCons.find(after);
    if (found != gCons.end()) {
      for (auto before : found->second) {
        if (!registered(before, after)) {
          opsAfterKey[before].push_back(after);
          ++nBeforeKey[after];
        }
      }
    }
  }

  auto readyToProcess = [&nIndicesAwaiting, &nBeforeKey](Op *op) {
    return (nIndicesAwaiting[op] == 0 && nBeforeKey[op] == 0);
  };

  // processing a tensor involves
  // reducing the counts in `awaiting' for
  // ops which use it, and detecting which
  // ops have nothing left to wait for as a
  // result of such updating.
  auto processTensor =
      [&opsToProcess, &nIndicesAwaiting, &readyToProcess](Tensor *tensor) {
        for (auto &op_count : tensor->consumers.getMap()) {
          Op *op = op_count.first;
          nIndicesAwaiting[op] -= op_count.second;
          if (readyToProcess(op)) {
            opsToProcess.push(op_count.first);
          }
        }
      };

  // we will start by processing
  // the tensors which have no producers
  auto t0 = tensors.getNoProducerIds();
  for (auto &id : t0) {
    processTensor(tensors.get(id));
  }

  while (!opsToProcess.empty()) {
    auto op = opsToProcess.top();
    opsToProcess.pop();
    sorted.push_back(op);
    for (Op *waitingOp : opsAfterKey[op]) {
      --nBeforeKey[waitingOp];
      if (readyToProcess(waitingOp)) {
        opsToProcess.push(waitingOp);
      }
    }

    for (auto &tensor_indices : op->output->indicesMap()) {
      processTensor(tensor_indices.first);
    }
  }

  return sorted;
}

} // namespace poponnx
