// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_VIRTUALGRAPH_HPP
#define GUARD_NEURALNET_VIRTUALGRAPH_HPP

#include <poplar/Graph.hpp>

namespace popart {
namespace popx {

class VirtualGraph {
public:
  VirtualGraph(poplar::Graph &&computeTilesGraph_)
      : computeTilesGraph(
            std::make_shared<poplar::Graph>(std::move(computeTilesGraph_))) {}
  VirtualGraph(poplar::Graph &&computeTilesGraph_,
               poplar::Graph &&ioTilesGraph_)
      : computeTilesGraph(
            std::make_shared<poplar::Graph>(std::move(computeTilesGraph_))),
        ioTilesGraph(
            std::make_shared<poplar::Graph>(std::move(ioTilesGraph_))) {}

  bool hasComputeTilesGraph() { return computeTilesGraph.get() != nullptr; }
  bool hasIoTilesGraph() { return ioTilesGraph.get() != nullptr; }

  poplar::Graph &getComputeTilesGraph() { return *computeTilesGraph; }
  poplar::Graph &getIoTilesGraph() { return *ioTilesGraph; }

private:
  std::shared_ptr<poplar::Graph> computeTilesGraph;
  std::shared_ptr<poplar::Graph> ioTilesGraph;
};

} // namespace popx
} // namespace popart

#endif
