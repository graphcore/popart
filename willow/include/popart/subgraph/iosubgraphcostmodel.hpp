// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_SUBGRAPH_IOSUBGRAPHCOSTMODEL_HPP_
#define POPART_WILLOW_INCLUDE_POPART_SUBGRAPH_IOSUBGRAPHCOSTMODEL_HPP_

#include <cstdint>
#include <map>
#include <memory>
#include <vector>

namespace popart {

class Op;

namespace outline {

class ValueCalculator {
public:
  // Class that memoises a cumulative outlining value for all schedule
  // locations to avoid repeated calculations.
  ValueCalculator(const std::vector<Op *> &schedule,
                  const std::map<Op *, int> &schedule_index);

  // Get the value for a range [start, end).
  float getValue(int64_t start, int64_t end) const;

private:
  // For each schedule location n, this stores the sum of values in the
  // range [0,n) -- e.g. not including n.
  std::vector<double> cumulativeValues;
};

class CopyCostCalculator {
public:
  // Class that memoises the copy cost for all schedule locations to avoid
  // repeated calculations.
  CopyCostCalculator(const std::vector<Op *> &schedule,
                     const std::map<Op *, int> &schedule_index);

  // Get the copy cost for a range [start, end).
  float getCopyCost(int64_t start, int64_t end) const;

private:
  // Assuming that outlining sections are defined by a range [start, last],
  // where the range includes the ‘last’ value (in contrast to the [start, end)
  // range where end is exclusive), we can create a vector for each possible
  // outlining section ‘start’ location. This vector comprises a container that
  // represents potential copy costs for input tensors. Each entry in the
  // container is a tuple that comprises two values: 1) the estimated copy costs
  // for that tensor and 2) the first ‘last’ value for which that copy cost
  // would apply (inclusive).
  std::vector<std::vector<std::tuple<float, int>>> inputCopyCost;

  // As above but for output copy cost for tensors. This stores for each 'last'
  // value a container of tuples with copy costs and last 'start' values for
  // which that copy cost applies.
  std::vector<std::vector<std::tuple<float, int>>> outputCopyCost;
};

class IoSubgraphCostModel {
public:
  // get a new cost estimate of a sub-sequence, using the cost of copying into
  // and out of cached subgraphs
  //
  // NOTE: We assume here that during the lifetime of IoSubgraphCostModel
  // the parameters schedule and schedule_index never change.
  float value(int64_t start,
              int64_t end,
              const std::vector<Op *> &schedule,
              const std::map<Op *, int> &schedule_index);

private:
  // Class to do value calculations.
  std::unique_ptr<ValueCalculator> valueCalculator;
  // Class to do copy cost calculations.
  std::unique_ptr<CopyCostCalculator> copyCostCalculator;
};

} // namespace outline
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_SUBGRAPH_IOSUBGRAPHCOSTMODEL_HPP_
