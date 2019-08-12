#include <popart/graph.hpp>
#include <popart/op.hpp>
#include <popart/patterns/sequenceexpander.hpp>
#include <popart/pbwrap.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>
#include <popart/topocons.hpp>

#include <algorithm>
#include <numeric>

namespace popart {

std::vector<const Tensor *> SequenceExpander::touches(Op *) const { return {}; }

static void connectPair(TensorId baseId,
                        std::unique_ptr<Op> &left,
                        std::unique_ptr<Op> &right) {
  auto tensor = PreAliasPattern::createIntermediateTensorId(baseId);
  logging::pattern::info("Adding intermediate tensor with id {}", tensor);

  left->createAndConnectOutTensor(0, tensor);
  right->connectInTensor(0, tensor);
}

bool SequenceExpander::expand(std::vector<std::unique_ptr<Op>> &seq,
                              Op *op) const {

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

    op->getGraph().topoCons->transfer(op, step.get());

    op->getGraph().moveIntoGraph(std::move(step));
  }

  // Delete the matched op
  logging::pattern::info("Removing op {}", op->str());

  op->disconnectAllInputs();
  op->getGraph().eraseOp(op->id);

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

} // namespace popart
