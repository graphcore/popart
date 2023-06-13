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

ValueCalculator::ValueCalculator(
    const std::vector<Op *> &schedule,
    const std::map<Op *, int> &schedule_index) :
  cumulativeValues(schedule.size()+1, 0.0f) {

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
  double cumulativeValue = 0.0;
  cumulativeValues[0] = cumulativeValue;
  for (auto i = 0; i < schedule.size(); ++i) {
    // in this cost model, we take the value to scale as sqrt(input size)
    cumulativeValue += schedule[i]->getSubgraphValue() *
                       std::sqrt(static_cast<float>(getInBytes(schedule[i])));
    cumulativeValues[i+1] = cumulativeValue;
  }
}

float ValueCalculator::getValue(int64_t start, int64_t end) const {
  if (start < end) {
    return static_cast<float>(cumulativeValues[end] - cumulativeValues[start]);
  } else {
    return 0.0f;
  }
}

CopyCostCalculator::CopyCostCalculator(
    const std::vector<Op *> &schedule,
    const std::map<Op *, int> &schedule_index) :
  inputCopyCost(schedule.size()),
  outputCopyCost(schedule.size()) {

  for (int i = 0; i < schedule.size(); ++i) {
    Op *op = schedule[i];

    for (auto tensor : op->input->tensors()) {
      float copyCost = 1.0f * std::sqrt(static_cast<float>(tensor->info.nbytes()));
      int producedAt = tensor->hasProducer()
        ? schedule_index.at(tensor->getProducerUnsafe())
        : -1;

      // For an outlining section that starts any time (strictly) after this
      // tensor is produced, but that does include this tensor's producing op,
      // we invoke a cost for copying the input argument.
      for (int start = producedAt + 1; start <= i; ++start) {
        inputCopyCost[start].push_back(std::make_tuple(copyCost, i));
      }
    }

    for (auto tensor : op->output->tensors()) {
      float copyCost = 1.0f * std::sqrt(static_cast<float>(tensor->info.nbytes()));
      const auto& consumerOps = tensor->consumers.getOps();
      if (consumerOps.size() > 0) {
        // Find latest schedule index where tensor is consumed.
        int lastConsumerAt = -1;
        for (auto consumer : consumerOps) {
          int consumerAt = schedule_index.at(consumer);
          if (consumerAt > lastConsumerAt) {
            lastConsumerAt = consumerAt;
          }
        }

        // If an outlining section has a ‘last’ value that falls between the
        // time when the tensor is produced (inclusive) and the time when it
        // is last consumed (exclusive), but starts no later than when the
        // tensor is produced, we invoke a copy cost for copying the output
        // tensor.
        for (int last = i; last < lastConsumerAt; ++last) {
          outputCopyCost[last].push_back(std::make_tuple(copyCost, i));
        }
      }
    }
  }
}

float CopyCostCalculator::getCopyCost(int64_t start, int64_t end) const {
  int64_t last = end -1;

  float totalCopyCost = 0.0f;
  // Add any applicable input copies.
  for (const auto& tup : inputCopyCost[start]) {
    const auto& firstLast = std::get<1>(tup);
    if (last >= firstLast) {
      const auto& copyCost = std::get<0>(tup);
      totalCopyCost += copyCost;
    }
  }
  // Add any applicable output copies.
  if (last >= 0) {
    for (const auto& tup : outputCopyCost[last]) {
      const auto& lastStart = std::get<1>(tup);
      if (start <= lastStart) {
        const auto& copyCost = std::get<0>(tup);
        totalCopyCost += copyCost;
      }
    }
  }
  return totalCopyCost;
}

float IoSubgraphCostModel::value(
    int64_t start,
    int64_t end,
    const std::vector<Op *> &schedule,
    const std::map<Op *, int> &schedule_index) {

  if (!valueCalculator) {
    // This calculation is done once.
    valueCalculator = std::make_unique<ValueCalculator>(schedule, schedule_index);
  }

  if (!copyCostCalculator) {
    // This calculation is done once.
    copyCostCalculator = std::make_unique<CopyCostCalculator>(schedule, schedule_index);
  }

  float sumValues = valueCalculator->getValue(start, end);
  float copyCost = copyCostCalculator->getCopyCost(start, end);

  sumValues -= copyCost;
  return sumValues;
}

} // namespace outline
} // namespace popart
