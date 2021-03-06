// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <transforms/autodiff/tensorgradregistry.hpp>

#include <popart/graph.hpp>
#include <popart/tensorindex.hpp>

namespace popart {

TensorGradRegistry::TensorGradRegistry(AutodiffIrInterface &ir_)
    : partial{}, complete{}, failed{}, ir{ir_}, expectedNumEdges{},
      edgesToLoss{} {}

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

  // Make complete even when it's 0.
  tryMakeComplete(nonGrad);
}

int TensorGradRegistry::getNumberExpectedEdges(Tensor *nonGrad) const {
  auto found = expectedNumEdges.find(nonGrad->id);
  if (found != expectedNumEdges.end()) {
    return found->second;
  } else {
    return edgesToLoss.at(nonGrad);
  }
}

void TensorGradRegistry::tryMakeComplete(Tensor *nonGrad) {

  const auto actualEdges   = partial[nonGrad->id].size();
  const auto expectedEdges = expectedNumEdges.at(nonGrad->id);

  const bool isComplete = (actualEdges >= expectedEdges);

  if (expectedEdges > 0) {
    logging::transform::trace("[Autodiff] Recorded edge gradient {}/{} is "
                              "available for '{}' ({})",
                              actualEdges,
                              expectedEdges,
                              nonGrad->id,
                              isComplete ? "complete" : "incomplete");

    if (isComplete) {
      // All edge gradients are available.
      complete[nonGrad->id] = partial[nonGrad->id];
      partial.erase(nonGrad->id);
    }

  } else {
    logging::transform::trace("[Autodiff] Recorded no edge gradients will be "
                              "available for '{}'",
                              nonGrad->id);

    // No edge gradients exist.
    failed[nonGrad->id] = partial[nonGrad->id];
    partial.erase(nonGrad->id);
  }
}

nonstd::optional<TensorGradRegistry::TMap::value_type>
TensorGradRegistry::popComplete() {
  if (!complete.empty()) {
    TMap::value_type first = *complete.begin();
    complete.erase(complete.begin());
    return first;
  } else {
    return nonstd::optional<TMap::value_type>();
  }
}

nonstd::optional<TensorGradRegistry::TMap::value_type>
TensorGradRegistry::popFailed() {
  if (!failed.empty()) {
    TMap::value_type first = *failed.begin();
    failed.erase(failed.begin());
    return first;
  } else {
    return nonstd::optional<TMap::value_type>();
  }
}

void TensorGradRegistry::initialize() {

  // set all edge counts to zero (we set from scratch in this function)
  for (TensorId tid : ir.get().getMainGraph().getTensors().getAllTensorIds()) {
    Tensor *t      = ir.get().getMainGraph().getTensors().get(tid);
    edgesToLoss[t] = 0;
  }

  for (auto &id_op : ir.get().getMainGraph().getOps()) {
    Op *op = id_op.second.get();

    // If Op goes to Loss, then for each of its inputs, +1 path
    if (op->toLoss == PathToLoss::Yes) {
      for (auto index_tensor : op->input->tensorMap()) {
        auto inTensor = index_tensor.second;
        ++edgesToLoss[inTensor];
      }
    }
  }

  for (TensorId tid : ir.get().getMainGraph().getTensors().getAllTensorIds()) {
    Tensor *t = ir.get().getMainGraph().getTensors().get(tid);
    logging::trace("Edges to loss: {} {}", tid, edgesToLoss[t]);
  }
}

void TensorGradRegistry::logDump(logging::Level level) const {

  auto logTMap = [&](const TMap &tmap, const char *store) {
    for (const auto &entry : tmap) {
      Tensor *nonGrad = ir.get().getMainGraph().getTensors().get(entry.first);
      logging::log(logging::Module::transform,
                   level,
                   logging::format("[Autodiff]  - {}/{} {} ({})",
                                   entry.second.size(),
                                   getNumberExpectedEdges(nonGrad),
                                   entry.first,
                                   store));
    }
  };

  logging::log(
      logging::Module::transform, level, "[Autodiff] TensorGradRegistry:");

  if (partial.empty() && complete.empty() && failed.empty()) {
    logging::log(logging::Module::transform, level, "[Autodiff]  - empty");
  } else {
    logTMap(partial, "partial");
    logTMap(complete, "complete");
    logTMap(failed, "failed");
  }
}

} // namespace popart