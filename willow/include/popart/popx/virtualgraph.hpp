// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_VIRTUALGRAPH_HPP
#define GUARD_NEURALNET_VIRTUALGRAPH_HPP

#include <snap/Graph.hpp>

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

#endif
