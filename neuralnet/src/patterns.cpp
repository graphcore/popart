#include <neuralnet/error.hpp>
#include <neuralnet/graph.hpp>
#include <neuralnet/pad.hpp>
#include <neuralnet/patterns.hpp>
#include <neuralnet/tensor.hpp>

namespace neuralnet {

bool Pattern::removesNoAnchored(const Op *op) const {
  for (auto &tensor : removes(op)) {
    if (op->pgraph->isAnchored(tensor->id)) {
      return false;
    }
  }
  return true;
};

bool PreUniRepl::matches(const Op *op) const {
  // op must have 1 input, and that input
  // must be consumed by only op (and only once)
  if (op->input.n() != 1) {
    return false;
  } else if (op->input.tensor(0)->consumers.getTotal() != 1) {
    return false;
  }

  // A sum with only one input
  else if (op->opType == OpType::SUM) {
    return true;
    // A pad with zero-padding
  } else if (op->opType == OpType::PAD &&
             dynamic_cast<const PadOp *>(op)->padSizeZero()) {
    return true;
  } else {
    return false;
  }
}

std::vector<const Tensor *> PreUniRepl::removes(const Op *op) const {
  return {op->input.tensor(0)};
}

// (see .hpp for ascii picture definitions)
void PreUniRepl::apply(Op *op) const {
  // op is []
  // ()
  Tensor *tensorIn = op->input.tensor(0);
  // (.)
  Tensor *tensorOut = op->output.tensor(0);
  // [.]
  auto op0 = tensorIn->getProducer();
  // (.) gets all consumers of () other than []
  tensorOut->consumers.extend(tensorIn->consumers.getMap());
  tensorOut->consumers.decrement(op);
  // [.] produces (.) directly
  int index = op0->output.indices(tensorIn)[0];
  op0->output.reset(index, tensorOut);
  tensorOut->resetProducer(op0);
  Graph *graph = op->pgraph;
  // delete ()
  graph->tensors.remove(tensorIn->id); // name);
  // delete [.]
  graph->eraseOp(op->id);
}

bool PostNRepl::matches(const Op *op) const {

  // Gradient of ADD
  if (op->opType == OpType::ADDGRAD) {
    // good so far
  }
  // A sum with only one input
  else if (op->opType == OpType::SUM && op->input.n() == 1) {
    // good so far
  }
  // A pad with zero-padding
  else if (op->opType == OpType::PAD &&
           dynamic_cast<const PadOp *>(op)->padSizeZero()) {
    // good so far
  } else {
    return false;
  }
  // we check that the consumer topological constraints
  // of (ori), and (rep1, rep2, rep3) can be be resolved
  // if [*] is removed.
  TopoBundle tcInf = getTopoConInfo(op);

  // merging weak constraints is impossible
  if (tcInf.nWeakTopoCons > 0) {
    return false;
  }

  // at most 1 consumer can be last
  else if (tcInf.nTopoLasts > 1) {
    return false;
  }

  // if the last consumer is op, that won't
  // work as op is going to be removed.
  // Also, this should not be possible if this
  // is really a replicating op
  else if (tcInf.lastCon == op) {
    return false;
  }

  // we have a viable match
  return true;
}

PostNRepl::TopoBundle PostNRepl::getTopoConInfo(const Op *op) const {
  TopoBundle tcInf;
  std::vector<const Tensor *> wouldMerge;
  // The unique input to op:
  wouldMerge.push_back(op->input.tensor(0));
  // And the N output ops:
  for (auto &t_inds : op->output.indicesMap()) {
    wouldMerge.push_back(t_inds.first);
  }

  for (auto &tensor : wouldMerge) {
    if (tensor->consumers.hasTopoLast()) {
      ++tcInf.nTopoLasts;
      tcInf.lastCon = tensor->consumers.getTopoLast();
    }
    if (tensor->consumers.hasWeakTopoCons()) {
      ++tcInf.nWeakTopoCons;
    }
  }
  return tcInf;
}

// removes all the outputs of the root op from the Graph
std::vector<const Tensor *> PostNRepl::removes(const Op *op) const {
  std::vector<const Tensor *> outs;
  for (auto &t_inds : op->output.indicesMap()) {
    outs.push_back(t_inds.first);
  }
  return outs;
}

// (see .hpp for ascii picture definitions)
void PostNRepl::apply(Op *op) const {

  // op is [*]
  Tensor *ori = op->input.tensor(0);

  // get the info on which will be the last op
  // to consume ori, if there is one
  TopoBundle tcInf = getTopoConInfo(op);

  std::vector<Tensor *> replicates;
  // setting replicates (rep1), (rep2), (rep3)
  for (auto &ind_t : op->output.tensorMap()) {
    replicates.push_back(ind_t.second);
  }

  for (auto t_repl : replicates) {
    // for rep1 : {[op0], [op2]}
    for (Op *op_z : t_repl->consumers.getOps()) {
      // at what indices is (rep1) consumed?
      for (int index : op_z->input.indices(t_repl)) {
        // must rather consume ori
        op_z->input.reset(index, ori);
      }
    }
    // ori is consumed by all consumers of t_repl
    // (this is the same wiring as above, always needs
    // to be done for tensor and op)
    ori->consumers.extend(t_repl->consumers.getMap());
  }
  ori->consumers.decrement(op);
  Graph *graph = op->pgraph;
  // delete replicates
  for (auto repl : replicates) {
    graph->tensors.remove(repl->id);
  }

  // delete [*]
  graph->eraseOp(op->id);

  // finally, clear up topo last if necessary
  if (ori->consumers.hasTopoLast()) {
    ori->consumers.removeTopoLast();
  }
  if (tcInf.nTopoLasts == 1) {
    ori->consumers.setTopoLast(tcInf.lastCon);
  }
}

} // namespace neuralnet
