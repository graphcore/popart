// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_VIRTUALGRAPH_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_VIRTUALGRAPH_HPP_

#include <memory>
#include <snap/Graph.hpp>
#include <utility>

namespace popart {
namespace popx {

class VirtualGraph {
public:
  VirtualGraph(snap::Graph &&computeTilesGraph_)
      : computeTilesGraph(
            std::make_shared<snap::Graph>(std::move(computeTilesGraph_))) {}
  VirtualGraph(snap::Graph &&computeTilesGraph_, snap::Graph &&ioTilesGraph_)
      : computeTilesGraph(
            std::make_shared<snap::Graph>(std::move(computeTilesGraph_))),
        ioTilesGraph(std::make_shared<snap::Graph>(std::move(ioTilesGraph_))) {}

  bool hasComputeTilesGraph() { return computeTilesGraph.get() != nullptr; }
  bool hasIoTilesGraph() { return ioTilesGraph.get() != nullptr; }

  snap::Graph &getComputeTilesGraph() { return *computeTilesGraph; }
  snap::Graph &getIoTilesGraph() { return *ioTilesGraph; }

private:
  std::shared_ptr<snap::Graph> computeTilesGraph;
  std::shared_ptr<snap::Graph> ioTilesGraph;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_VIRTUALGRAPH_HPP_
