// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_SUFFIXTREE_HPP
#define GUARD_NEURALNET_SUFFIXTREE_HPP

#include "match.hpp"
#include <vector>

namespace fwtools {
namespace subgraph {
namespace suffixtree {

std::vector<Match> getInternal(const std::vector<int> &s);

} // namespace suffixtree
} // namespace subgraph
} // namespace fwtools

#endif
