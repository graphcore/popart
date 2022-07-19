// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_SUBGRAPH_SUFFIXTREE_HPP_
#define POPART_WILLOW_INCLUDE_POPART_SUBGRAPH_SUFFIXTREE_HPP_

#include <vector>

namespace fwtools {
namespace subgraph {
class Match;

namespace suffixtree {

std::vector<Match> getInternal(const std::vector<int> &s);

} // namespace suffixtree
} // namespace subgraph
} // namespace fwtools

#endif // POPART_WILLOW_INCLUDE_POPART_SUBGRAPH_SUFFIXTREE_HPP_
