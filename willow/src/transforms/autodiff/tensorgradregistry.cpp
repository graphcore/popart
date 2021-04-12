// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <transforms/autodiff/tensorgradregistry.hpp>

#include <popart/graph.hpp>
#include <popart/tensorindex.hpp>

namespace popart {

void TensorGradRegistry::insert(Tensor *nonGrad, Tensor *grad) {
  // The expected number of edges is assumed to be the same as the
  // number of edges to the loss for the non-grad tensor.
  if (expectedNumEdges.find(nonGrad->id) == expectedNumEdges.end()) {
    expectedNumEdges.insert({nonGrad->id, edgesToLoss.at(nonGrad)});
  }

  auto found = partial.find(nonGrad->id);
  if (found == partial.end()) {
    partial.insert({nonGrad->id, {grad}});
  } else {
    partial[nonGrad->id].push_back(grad);
  }

  tryMakeComplete(nonGrad);
}

void TensorGradRegistry::decrementNumberExpectedEdges(Tensor *nonGrad) {
  auto found = expectedNumEdges.find(nonGrad->id);
  if (found == expectedNumEdges.end()) {
    expectedNumEdges.insert({nonGrad->id, edgesToLoss.at(nonGrad) - 1});
  } else {
    found->second--;
  }

  // Only make complete if this is already in partials.
  // This prevents adding entries with 0 gradient edges.
  if (partial.find(nonGrad->id) != partial.end()) {
    tryMakeComplete(nonGrad);
  }
}

int TensorGradRegistry::getNumberExpectedEdges(Tensor *nonGrad) {
  auto found = expectedNumEdges.find(nonGrad->id);
  if (found != expectedNumEdges.end()) {
    return found->second;
  } else {
    return edgesToLoss.at(nonGrad);
  }
}

void TensorGradRegistry::tryMakeComplete(Tensor *nonGrad) {
  if (partial[nonGrad->id].size() == expectedNumEdges.at(nonGrad->id)) {
    complete[nonGrad->id] = partial[nonGrad->id];
    partial.erase(nonGrad->id);
  }
}

std::map<TensorId, std::vector<Tensor *>> TensorGradRegistry::popComplete() {
  auto toRet = complete;
  complete   = {};
  return toRet;
}

void TensorGradRegistry::initialize(AutodiffIrInterface &ir) {

  // set all edge counts to zero (we set from scratch in this function)
  for (TensorId tid : ir.getMainGraph().getTensors().getAllTensorIds()) {
    Tensor *t      = ir.getMainGraph().getTensors().get(tid);
    edgesToLoss[t] = 0;
  }

  for (auto &id_op : ir.getMainGraph().getOps()) {
    Op *op = id_op.second.get();

    // If Op goes to Loss, then for each of its inputs, +1 path
    if (op->toLoss == PathToLoss::Yes) {
      for (auto index_tensor : op->input->tensorMap()) {
        auto inTensor = index_tensor.second;
        ++edgesToLoss[inTensor];
      }
    }
  }

  for (TensorId tid : ir.getMainGraph().getTensors().getAllTensorIds()) {
    Tensor *t = ir.getMainGraph().getTensors().get(tid);
    logging::trace("Edges to loss: {} {}", tid, edgesToLoss[t]);
  }
}

} // namespace popart