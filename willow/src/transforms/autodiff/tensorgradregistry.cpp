// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <transforms/autodiff/tensorgradregistry.hpp>

namespace popart {

void TensorGradRegistry::insert(Tensor *nonGrad, Tensor *grad) {
  // The expected number of edges is assumed to be the same as the
  // number of edges to the loss for the non-grad tensor.
  if (expectedNumEdges.find(nonGrad->id) == expectedNumEdges.end()) {
    expectedNumEdges.insert({nonGrad->id, nonGrad->nEdgesToLoss});
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
    expectedNumEdges.insert({nonGrad->id, nonGrad->nEdgesToLoss - 1});
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
    return nonGrad->nEdgesToLoss;
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

} // namespace popart