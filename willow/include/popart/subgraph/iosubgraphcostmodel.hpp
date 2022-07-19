// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_SUBGRAPH_IOSUBGRAPHCOSTMODEL_HPP_
#define POPART_WILLOW_INCLUDE_POPART_SUBGRAPH_IOSUBGRAPHCOSTMODEL_HPP_

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

#endif // POPART_WILLOW_INCLUDE_POPART_SUBGRAPH_IOSUBGRAPHCOSTMODEL_HPP_
