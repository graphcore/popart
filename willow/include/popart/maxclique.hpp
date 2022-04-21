// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_MAXCLIQUE_HPP
#define GUARD_NEURALNET_MAXCLIQUE_HPP

#include <memory>
#include <vector>

namespace popart {
namespace graphclique {

using ColorGroup  = std::vector<int>;
using ColorGroups = std::vector<ColorGroup>;
using Vertex      = std::pair<int, int>;
using Vertices    = std::vector<Vertex>;

class AGraphImpl;
class AGraph {
public:
  AGraph(int size_);
  AGraph(AGraph const &other);
  AGraph() = delete;
  ~AGraph();
  void addEdge(int i, int j);
  bool getEdge(int i, int j) const;
  int numVertices() const;

private:
  std::unique_ptr<AGraphImpl> pimpl;
};

class MaxClique {
public:
  // Initialize MaxClique with an edge-graph
  MaxClique(AGraph graph);
  ~MaxClique() {}

  // Get up to maxCount maximum cliques of at least minSize nodes in descending
  // order. The algorithm will repeatedly try to find the maximum clique first,
  // remove the nodes of that clique and find the next largest maximum clique.
  std::vector<std::vector<int>>
  getMaximumCliques(int minSize, int maxCount, const float stepLimit = 0.03f);

  bool getEdge(int i, int j) const;

  bool colorCut(const int, const ColorGroup &);
  void verticesCut(const Vertices &, Vertices &);

  void setColorsToDegrees(Vertices &);
  void sortByColor(Vertices &);

  void updateDegree(Vertices &);
  void sortByDegree(Vertices &);

  void branch(Vertices, int depth);

private:
  AGraph agraph_;
  int numVerices_;
  Vertices vertices_;
  ColorGroups colors_;
  ColorGroup groupColor_;
  ColorGroup maxGroupColor_;
  int steps_;
  float stepLimit_;
  std::vector<std::pair<int, int>> stepCount_;
};

} // namespace graphclique
} // namespace popart

#endif
