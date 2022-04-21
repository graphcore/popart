// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "popart/logging.hpp"
#include <transforms/autodiff/opgradregistry.hpp>

#include <popart/graph.hpp>
#include <popart/op.hpp>
#include <popart/tensorindex.hpp>

namespace popart {

OpGradRegistry::OpGradRegistry(Graph &fwdGraph_)
    : fwdGraph{fwdGraph_}, partial{}, complete{}, failed{}, completeOrFailed{},
      edgesToLoss{} {}

void OpGradRegistry::insert(Op *nonGrad, int index) {

  if (nonGrad->outTensor(index)->toLoss == PathToLoss::Yes) {

    auto completeOrFailedIt = completeOrFailed.find(nonGrad->id);
    if (completeOrFailedIt == completeOrFailed.end()) {

      auto partialIt = partial.find(nonGrad->id);
      // so far NO gradients for nonGrad are in:
      if (partialIt == partial.end()) {
        partial.insert({nonGrad->id, {}});
      }
      // this should be removed when we're happy the IL (internal logic)
      // is correct:
      if (partial[nonGrad->id].count(index) != 0) {
        throw internal_error("index already present in OpGradRegistry::insert");
      }

      partial[nonGrad->id].insert(index);

      // Check whether an Op is ready to grow gradients based on the set of
      // output indices of `op' for which a gradient is available. Currently,
      // this will just compare the number of tensors for which we have a
      // gradient that are on the path to loss with expected number of paths to
      // final loss.
      if (edgesToLoss.at(nonGrad) == partial[nonGrad->id].size()) {
        complete.push_back(nonGrad);
        completeOrFailed.insert(nonGrad->id);
        partial.erase(nonGrad->id);
      }
    } else {
      throw internal_error("[Autodiff] Unable to process gradient for {} "
                           "because it has already failed or completed",
                           nonGrad->str());
    }
  } else {
    logging::transform::trace("[Autodiff] Ignoring availability of gradient "
                              "of output '{}' of {} because it is not "
                              "on the path to loss",
                              nonGrad->outId(index),
                              nonGrad->str());
  }
}

void OpGradRegistry::fail(Op *nonGrad, int index) {

  if (nonGrad->outTensor(index)->toLoss == PathToLoss::Yes) {

    auto completeOrFailedIt = completeOrFailed.find(nonGrad->id);
    if (completeOrFailedIt == completeOrFailed.end()) {
      auto found = partial.find(nonGrad->id);
      if (found == partial.end()) {
        partial.erase(nonGrad->id);
      }
      failed.push_back(nonGrad);
      completeOrFailed.insert(nonGrad->id);
    } else {
      logging::transform::trace("[Autodiff] Unable to fail {} due to the "
                                "failure to grow gradient of output '{}' "
                                "because it has already failed or completed",
                                nonGrad->str(),
                                nonGrad->outId(index));
    }

  } else {
    logging::transform::trace("[Autodiff] Ignoring failure to grow gradient "
                              "of output '{}' of {} because it is not "
                              "on the path to loss",
                              nonGrad->outId(index),
                              nonGrad->str());
  }
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
  for (auto &id_op : fwdGraph.get().getOps()) {
    Op *op          = id_op.second.get();
    edgesToLoss[op] = 0;
  }

  for (auto &id_op : fwdGraph.get().getOps()) {
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
      Op *nonGrad = fwdGraph.get().getOp(entry.first);
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
