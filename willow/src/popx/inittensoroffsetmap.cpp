// Copyright (c) 2023 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <cstddef>
#include <map>
#include <vector>
#include <poplar/Graph.hpp>
#include <poplar/Target.hpp>
#include <popart/popx/inittensoroffsetmap.hpp>

namespace popart {
namespace popx {

std::size_t InitTensorOffsetMap::getOffset(poplar::Graph &graph) {
  auto findIt = offsets.find(&graph);

  if (findIt == offsets.end()) {
    offsets.insert({&graph, 0});
    return 0;
  } else {
    return findIt->second;
  }
}

void InitTensorOffsetMap::setOffset(poplar::Graph &graph,
                                    const std::size_t offset) {
  offsets[&graph] = offset;
}

} // namespace popx
} // namespace popart
