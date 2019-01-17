#include <poponnx/ir.hpp>
#include <poponnx/op.hpp>
#include <poponnx/patterns/sequenceexpander.hpp>
#include <poponnx/pbwrap.hpp>
#include <poponnx/tensor.hpp>

#include <algorithm>
#include <numeric>

namespace poponnx {

std::vector<const Tensor *> SequenceExpander::touches(Op *) const { return {}; }

static void connectPair(TensorId baseId,
                        std::unique_ptr<Op> &left,
                        std::unique_ptr<Op> &right) {
  auto tensor = Pattern::createIntermediateTensorId(baseId);
  logging::pattern::info("Adding intermediate tensor with id {}", tensor);

  left->createAndConnectOutTensor(0, tensor);
  right->connectInTensor(0, tensor);
}

bool SequenceExpander::expand(std::vector<std::unique_ptr<Op>> &seq,
                              Op *op) const {
  auto ir = op->pir;

  // Connect the input tensors to the front of the sequence
  auto &front = seq.front();
  for (auto &tensors : op->input->tensorIdMap()) {
    front->connectInTensor(tensors.first, tensors.second);
  }

  // Connect the output tensors to the back of the sequence
  auto &back = seq.back();
  for (auto &tensors : op->output->tensorIdMap()) {
    back->connectOutTensor(tensors.first, tensors.second);
  }

  // Connect the sequence of ops with intermediate tensors
  for (int i = 0; i < seq.size() - 1; ++i) {
    connectPair(op->input->id(0), seq[i], seq[i + 1]);
  }

  // Add the ops into the IR
  for (auto &step : seq) {
    logging::pattern::info("Inserting op {}", step->str());
    step->setup();
    ir->moveIntoIr(std::move(step));
  }

  // Delete the matched op
  logging::pattern::info("Removing op {}", op->str());

  op->disconnectAllInputs();
  ir->eraseOp(op->id);

  return true;
}

bool SequenceExpander::apply(Op *op) const {
  auto seq = sequence(op);

  if (seq.size() > 0) {
    return expand(seq, op);
  } else {
    throw error("No ops returned to replace {} in SequenceExpander", op->str());
  }
}

} // namespace poponnx
