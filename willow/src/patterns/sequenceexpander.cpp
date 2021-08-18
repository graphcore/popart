// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <popart/graph.hpp>
#include <popart/ir.hpp>
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

  // This code is a little awkward here due to scoping. Currently baseId may
  // already have a scope applied but createAndConnectOutTensor will add left's
  // scope to the tensor ID as well but connectInTensor will not. I think the
  // correct way of dealing with this is by re-basing the ID's scope to left's
  // scope by stripping out baseId's scope. This probably could be cleaner.
  TensorId unscopedBaseId = baseId;
  auto findScope          = baseId.str().find_last_of('/');
  if (findScope != std::string::npos) {
    unscopedBaseId = baseId.str().substr(findScope + 1);
  }

  auto tensor = left->getIr().createIntermediateTensorId(unscopedBaseId);
  logging::pattern::info("Adding intermediate tensor with id {}", tensor);

  // The createAndConnectOutTensor already applies scope but connectInTensor
  // does not.
  left->createAndConnectOutTensor(0, tensor);
  right->connectInTensor(0, (left->getScope() / tensor.str()).str());
}

bool SequenceExpander::expand(std::vector<std::unique_ptr<Op>> &seq,
                              Op *op) const {

  // An unwritten assumption in connectPair is that all ops in the seq vector
  // have the same scope as the op parameter. We check this precondition here to
  // catch bugs early.
  for (const auto &seqOpPtr : seq) {
    if (seqOpPtr->getScope() != op->getScope()) {
      throw error("Sequence Op ({}) does not match the scope of Op it is "
                  "replacing ({}) in SequenceExpander",
                  seqOpPtr->str(),
                  op->str());
    }
  }

  auto inputMap  = op->input->tensorIdMap();
  auto outputMap = op->output->tensorIdMap();
  op->disconnectAllInputs();
  op->disconnectAllOutputs();

  // Connect the input tensors to the front of the sequence
  auto &front = seq.front();
  for (auto &tensors : inputMap) {
    front->connectInTensor(tensors.first, tensors.second);
  }

  // Connect the output tensors to the back of the sequence
  auto &back = seq.back();
  for (auto &tensors : outputMap) {
    back->connectOutTensor(tensors.first, tensors.second);
  }

  // Connect the sequence of ops with intermediate tensors
  for (int i = 0; i < seq.size() - 1; ++i) {
    connectPair(inputMap.at(0), seq[i], seq[i + 1]);
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
