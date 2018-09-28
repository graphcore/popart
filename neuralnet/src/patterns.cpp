#include <neuralnet/error.hpp>
#include <neuralnet/graph.hpp>
#include <neuralnet/pad.hpp>
#include <neuralnet/patterns.hpp>
#include <neuralnet/tensor.hpp>

namespace neuralnet {

bool Pattern::removesNoAnchored(const Op *) const {
  // TODO correct this
  return true;
};

bool Identity::matches(const Op *op) const {
  // A sum with only one input
  if (op->opType == OpType::SUM && op->input.n() == 1) {
    return true;
    // A pad with zero-padding
  } else if (op->opType == OpType::PAD &&
             dynamic_cast<const PadOp *>(op)->padSizeZero()) {
    return true;
  } else {
    return false;
  }
}

std::vector<const Tensor *> Identity::removes(const Op *op) const {
  return {op->input.tensor(0)};
}

// (see .hpp for ascii picture definitions)
void Identity::apply(Op *op) const {
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

} // namespace neuralnet
