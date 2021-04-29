// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "popart/logging.hpp"
#include <transforms/autodiff/opgradregistry.hpp>

#include <popart/graph.hpp>
#include <popart/op.hpp>
#include <popart/tensorindex.hpp>

namespace popart {

OpGradRegistry::OpGradRegistry(AutodiffIrInterface &ir_)
    : ir{ir_}, partial{}, complete{}, failed{}, edgesToLoss{} {}

void OpGradRegistry::insert(Op *nonGrad, int index) {

  auto found = partial.find(nonGrad->id);
  // so far NO gradients for nonGrad are in:
  if (found == partial.end()) {
    partial.insert({nonGrad->id, {}});
  }
  // this should be removed when we're happy the IL (internal logic)
  // is correct:
  if (partial[nonGrad->id].count(index) != 0) {
    throw internal_error("index already present in OpGradRegistry::insert");
  }

  partial[nonGrad->id].insert(index);

  // Check whether an Op is ready to grow gradients based on the set of output
  // indices of `op' for which a gradient is available. Currently, this will
  // just compare the size of the set passed in with number of paths to final
  // loss.
  if (edgesToLoss.at(nonGrad) == partial[nonGrad->id].size()) {
    complete.push_back(nonGrad);
    partial.erase(nonGrad->id);
  }
}

void OpGradRegistry::fail(Op *nonGrad) {
  auto found = partial.find(nonGrad->id);
  if (found == partial.end()) {
    partial.erase(nonGrad->id);
  }
  failed.push_back(nonGrad);
}

nonstd::optional<Op *> OpGradRegistry::popComplete() {
  if (!complete.empty()) {
    Op *nonGrad = complete.front();
    complete.pop_front();
    return nonGrad;
  } else {
    return nonstd::optional<Op *>();
  }
}

nonstd::optional<Op *> OpGradRegistry::popFailed() {
  if (!failed.empty()) {
    Op *nonGrad = failed.front();
    failed.pop_front();
    return nonGrad;
  } else {
    return nonstd::optional<Op *>();
  }
}

void OpGradRegistry::initialize() {

  // set all edge counts to zero (we set from scratch in this function)
  for (auto &id_op : ir.get().getMainGraph().getOps()) {
    Op *op          = id_op.second.get();
    edgesToLoss[op] = 0;
  }

  for (auto &id_op : ir.get().getMainGraph().getOps()) {
    Op *op = id_op.second.get();

    // For each Op, how many OutIndices lead to loss?
    for (auto index_tensor : op->output->tensorMap()) {
      auto outTensor = index_tensor.second;
      if (outTensor->toLoss == PathToLoss::Yes) {
        ++edgesToLoss[op];
      }
    }
  }
}

void OpGradRegistry::logDump(logging::Level level) const {

  auto logMap = [&](const std::map<OpId, std::set<int>> &map,
                    const char *store) {
    for (const auto &entry : map) {
      Op *nonGrad = ir.get().getMainGraph().getOp(entry.first);
      logging::log(logging::Module::transform,
                   level,
                   logging::format("[Autodiff]  - {}/{} {} ({})",
                                   entry.second.size(),
                                   edgesToLoss.at(nonGrad),
                                   nonGrad->str(),
                                   store));
    }
  };

  auto logVec = [&](const std::list<Op *> &vec, const char *store) {
    for (const auto &nonGrad : vec) {
      logging::log(logging::Module::transform,
                   level,
                   logging::format("[Autodiff]  - {}/{} {} ({})",
                                   edgesToLoss.at(nonGrad),
                                   edgesToLoss.at(nonGrad),
                                   nonGrad->str(),
                                   store));
    }
  };

  logging::log(logging::Module::transform, level, "[Autodiff] OpGradRegistry:");

  if (partial.empty() && complete.empty() && failed.empty()) {
    logging::log(logging::Module::transform, level, "[Autodiff]  - empty");
  } else {
    logMap(partial, "partial");
    logVec(complete, "complete");
    logVec(failed, "failed");
  }
}

} // namespace popart
