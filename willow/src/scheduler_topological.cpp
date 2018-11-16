#include <algorithm>
#include <queue>
#include <poponnx/error.hpp>
#include <poponnx/ir.hpp>
#include <poponnx/scheduler_topological.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/util.hpp>

namespace willow {

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
TopologicalScheduler::getSchedule(const OpMap &ops,
                                  const class Tensors &tensors) const {
  // the topological sorting (to construct in this function)
  std::vector<Op *> sorted;
  // ops which have all their input tensors
  // created, and are not waiting for any ops
  // to run before them
  // OpPriorityComparer opCompare;
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
  // external constraints.
  // (1) map for each op to the number of ops which still
  // must be inserted before it can it can be inserted
  std::map<Op *, int> nOpsAwaiting;
  // (2) map from each op to a list of ops which are
  // waiting for it
  std::map<Op *, std::vector<Op *>> isWaitingFor;
  // initialise (1) and (2)
  for (auto &id_op : ops) {
    Op *op           = id_op.second.get();
    nOpsAwaiting[op] = 0;
    isWaitingFor[op] = {};
  }
  for (auto &id_op : ops) {
    Op *op = id_op.second.get();
    for (auto &tensor_indices : op->input.indicesMap()) {
      Tensor *inTen = tensor_indices.first;
      // which consumer(s) of inTens must appear before op?
      for (Op *otherCon : inTen->consumers.consumersWhichTopoBefore(op)) {
        if (std::find(isWaitingFor[otherCon].begin(),
                      isWaitingFor[otherCon].end(),
                      op) == isWaitingFor[otherCon].end()) {
          isWaitingFor[otherCon].push_back(op);
          ++nOpsAwaiting[op];
        }
      }
    }
  }

  auto readyToProcess = [&nIndicesAwaiting, &nOpsAwaiting](Op *op) {
    return (nIndicesAwaiting[op] == 0 && nOpsAwaiting[op] == 0);
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
    for (Op *waitingOp : isWaitingFor[op]) {
      --nOpsAwaiting[waitingOp];
      if (readyToProcess(waitingOp)) {
        opsToProcess.push(waitingOp);
      }
    }

    for (auto &tensor_indices : op->output.indicesMap()) {
      processTensor(tensor_indices.first);
    }
  }

  if (sorted.size() != ops.size()) {
    throw error("failure to sort topologically");
  }
  return sorted;
}

} // namespace willow
