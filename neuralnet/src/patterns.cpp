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
    return true;
  }
  // A sum with only one input
  else if (op->opType == OpType::SUM && op->input.n() == 1) {
    return true;
    // A pad with zero-padding
  } else if (op->opType == OpType::PAD &&
             dynamic_cast<const PadOp *>(op)->padSizeZero()) {
    return true;
  } else {
    return false;
  }
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
}

} // namespace neuralnet
