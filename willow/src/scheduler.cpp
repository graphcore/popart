#include <algorithm>
#include <queue>

#include <poponnx/ir.hpp>
#include <poponnx/scheduler.hpp>
#include <poponnx/tensor.hpp>

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
  bool operator()(Op *const &op1, Op *const &op2) const {
    return op1->priority < op2->priority;
  }
};

std::vector<Op *>
Scheduler::getPartialOpSchedule(const OpsBeforeKey &gCons) const {

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
    nIndicesAwaiting[op] = op->input.n();
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

    // we first check the local constraints on the consumers of "after"
    for (auto &tensor_indices : after->input.indicesMap()) {
      Tensor *inTen = tensor_indices.first;
      // which consumer(s) of inTens must appear before op?
      for (Op *before : inTen->consumers.consumersWhichTopoBefore(after)) {
        if (!registered(before, after)) {
          opsAfterKey[before].push_back(after);
          ++nBeforeKey[after];
        }
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

    for (auto &tensor_indices : op->output.indicesMap()) {
      processTensor(tensor_indices.first);
    }
  }

  return sorted;
}

} // namespace poponnx
