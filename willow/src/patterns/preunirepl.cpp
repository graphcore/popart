#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/op/pad.hpp>
#include <popart/patterns/patterns.hpp>
#include <popart/patterns/preunirepl.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>
#include <popart/tensors.hpp>

namespace popart {

bool PreUniRepl::matches(Op *op) const {
  // op must have 1 input, and that input
  // must have a producer and
  // must be consumed by only op (and only once)
  if (op->input->n() != 1) {
    return false;
  } else if (!op->inTensor(0)->hasProducer()) {
    return false;
  } else if (op->input->tensor(0)->consumers.getTotal() != 1) {
    return false;
  }

  // A sum with only one input
  else if (op->opid == Onnx::Operators::Sum_6 ||
           op->opid == Onnx::Operators::Sum_8) {
    return true;
    // A pad with zero-padding
  } else if (op->opid == Onnx::Operators::Pad_2 &&
             dynamic_cast<const PadOp *>(op)->padSizeZero()) {
    return true;
  } else {
    return false;
  }
}

std::vector<const Tensor *> PreUniRepl::touches(Op *op) const {
  return {op->input->tensor(0)};
}

// (see .hpp for ascii picture definitions)
bool PreUniRepl::apply(Op *op) const {
  // op is []
  // ()
  Tensor *tensorIn = op->input->tensor(0);
  // (.)
  Tensor *tensorOut = op->output->tensor(0);
  // [.]
  auto op0 = tensorIn->getProducer();
  // (.) gets all consumers of () other than []
  tensorOut->consumers.extend(tensorIn->consumers.getMap());
  tensorOut->consumers.decrement(op);
  // [.] produces (.) directly
  int index = op0->output->indices(tensorIn)[0];
  op0->output->reset(index, tensorOut);
  tensorOut->resetProducer(op0);
  Graph &graph = op->getGraph();
  // delete ()
  graph.getTensors().remove(tensorIn->id); // name);
  // delete [.]
  graph.eraseOp(op->id);

  return true;
}

namespace {
static PatternCreator<PreUniRepl>
    PreUniReplPattern(PreAliasPatternType::PREUNIREPL, "PreUniRepl");
}

} // namespace popart
