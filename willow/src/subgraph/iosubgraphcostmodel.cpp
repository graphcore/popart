// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <cmath>
#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <utility>
#include <vector>
#include <popart/op.hpp>
#include <popart/subgraph/iosubgraphcostmodel.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>

#include "popart/tensorinfo.hpp"

namespace popart {
namespace outline {

float IoSubgraphCostModel::value(
    int64_t start,
    int64_t end,
    const std::vector<Op *> &schedule,
    const std::map<Op *, int> &schedule_index) const {

  // total number of input bytes
  auto getInBytes = [](Op *op) {
    int64_t inBytes = 0;
    for (auto x : op->input->tensorMap()) {
      inBytes += x.second->info.nbytes();
    }
    return inBytes;
  };

  // the value of caching an all ops
  // (before adjusting for the cost of the copies)
  float sumValues = 0;
  for (auto i = start; i < end; ++i) {
    // in this cost model, we take the value to scale as sqrt(input size)
    sumValues += schedule[i]->getSubgraphValue() *
                 std::sqrt(static_cast<float>(getInBytes(schedule[i])));
  }

  // the external inputs and outputs, which we use to adjust the value of
  // match downwards
  std::set<Tensor *> externalInputs;
  std::set<Tensor *> externalOutputs;

  for (auto i = start; i < end; ++i) {
    Op *op = schedule[i];

    // externalInputs
    for (auto tensor : op->input->tensors()) {
      if (!tensor->hasProducer() ||
          schedule_index.at(tensor->getProducerUnsafe()) < start) {
        externalInputs.emplace(tensor);
      }
    }

    // externalOutputs
    for (auto tensor : op->output->tensors()) {
      for (auto consumer : tensor->consumers.getOps()) {
        if (schedule_index.at(consumer) >= end) {
          externalOutputs.insert(tensor);
        }
      }
    }
  }

  float copyCost = 0.0f;
  for (auto t : externalInputs) {
    // again, we make the cost scale as sqrt(input size)
    // we take coefficient 1.0f, a value intermediate of getSubgraphValue()
    // for valuable (like conv) ops and others (like relu)
    copyCost += 1.0f * std::sqrt(static_cast<float>(t->info.nbytes()));
  }
  for (auto t : externalOutputs) {
    copyCost += 1.0f * std::sqrt(static_cast<float>(t->info.nbytes()));
  }

  sumValues -= copyCost;
  return sumValues;
}

} // namespace outline
} // namespace popart
