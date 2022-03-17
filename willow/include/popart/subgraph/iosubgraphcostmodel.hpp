// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_IOSUBGRAPHCOSTMODEL_HPP
#define GUARD_NEURALNET_IOSUBGRAPHCOSTMODEL_HPP

#include <cstdint>
#include <map>
#include <vector>

namespace popart {

class Op;

namespace outline {

class IoSubgraphCostModel {

public:
  // get a new cost estimate of a sub-sequence, using the cost of copying into
  // and out of cached subgraphs
  float value(int64_t start,
              int64_t end,
              const std::vector<Op *> &schedule,
              const std::map<Op *, int> &schedule_index) const;
};

} // namespace outline
} // namespace popart

#endif
