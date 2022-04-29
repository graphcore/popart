// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <memory>
#include <transforms/autodiff/opgradregistry.hpp>
#include <utility>
#include <popart/graph.hpp>
#include <popart/op.hpp>
#include <popart/tensorindex.hpp>

#include "popart/error.hpp"
#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/tensor.hpp"
#include "popart/vendored/optional.hpp"
#include "popart/vertex.hpp"

namespace popart {

OpGradRegistry::OpGradRegistry(Graph &fwdGraph_)
    : fwdGraph{fwdGraph_}, partial{}, complete{}, failed{}, completeOrFailed{},
      edgesToLoss{} {}

// When we insert for the first time, bool helper in partial is
// set to same value as argument isProvided.
// If we insert and index is already in the set we handle logic as:
// A) if isProvided is true:
//   Do nothing. Allow potential overregistration when tensor is also one of
//   the provided gradients to autograd.
// B) if isProvided is false:
//   B1) if bool helper in partial is true set it to false.
//   B2) if bool helper in partial is false throw "index already present error".
void OpGradRegistry::insert(Op *nonGrad, int index, bool isProvided) {

  if (nonGrad->outTensor(index)->toLoss == PathToLoss::Yes) {

    auto completeOrFailedIt = completeOrFailed.find(nonGrad->id);
    if (completeOrFailedIt == completeOrFailed.end()) {

      auto partialIt = partial.find(nonGrad->id);
      // so far NO gradients for nonGrad are in:
      if (partialIt == partial.end()) {
        partial.insert({nonGrad->id, {}});
      }

      auto comp = [index](const std::pair<int, bool> &p) {
        return p.first == index;
      };
      auto indexProvided = std::find_if(
          begin(partial[nonGrad->id]), end(partial[nonGrad->id]), comp);
      if (indexProvided == partial[nonGrad->id].end()) {
        partial[nonGrad->id].insert(std::make_pair(index, isProvided));
      } else {
        if (!isProvided) {
          if (indexProvided->second) {
            partial[nonGrad->id].erase(std::make_pair(index, true));
            partial[nonGrad->id].insert(std::make_pair(index, false));
          } else {
            throw internal_error(
                "index already present in OpGradRegistry::insert");
          }
        }
      }

      // Check whether an Op is ready to grow gradients based on the set of
      // output indices of `op' for which a gradient is available. Currently,
      // this will just compare the number of tensors for which we have a
      // gradient that are on the path to loss with expected number of paths to
      // final loss.
      if (edgesToLoss.at(nonGrad) == partial[nonGrad->id].size()) {
        complete.push_back(nonGrad);
        completeOrFailed.insert(nonGrad->id);
      }
    } else {
      if (!isProvided) {
        auto comp2 = [index](const std::pair<int, bool> &p) {
          return p.first == index && p.second == true;
        };
        auto indexProvided = std::find_if(
            begin(partial[nonGrad->id]), end(partial[nonGrad->id]), comp2);
        if (indexProvided != partial[nonGrad->id].end()) {
          partial[nonGrad->id].erase(std::make_pair(index, true));
          partial[nonGrad->id].insert(std::make_pair(index, false));
        } else {
          throw internal_error("[Autodiff] Unable to process gradient for {} "
                               "because it has already failed or completed",
                               nonGrad->str());
        }
      }
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

  auto logMap = [&](const std::map<OpId, std::set<std::pair<int, bool>>> &map,
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
