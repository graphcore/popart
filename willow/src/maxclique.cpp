// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <boost/graph/adjacency_matrix.hpp>
#include <boost/graph/undirected_graph.hpp>
#include <memory>
#include <utility>
#include <vector>
#include <popart/error.hpp>
#include <popart/logging.hpp>
#include <popart/maxclique.hpp>

namespace popart {
namespace graphclique {

class AGraphImpl : public boost::adjacency_matrix<boost::undirectedS> {
public:
  AGraphImpl(int size_) : boost::adjacency_matrix<boost::undirectedS>(size_) {}
  AGraphImpl(AGraphImpl const &other)
      : boost::adjacency_matrix<boost::undirectedS>(other) {}
};

AGraph::AGraph(int size_) : pimpl(new AGraphImpl(size_)) {}

AGraph::AGraph(AGraph const &other)
    : pimpl(std::make_unique<AGraphImpl>(*other.pimpl)) {}

AGraph::~AGraph() {}

void AGraph::addEdge(int i, int j) { boost::add_edge(i, j, *pimpl); }

int AGraph::numVertices() const {
  return static_cast<int>(boost::num_vertices(*pimpl));
}

bool AGraph::getEdge(int i, int j) const { return pimpl->get_edge(i, j); }

MaxClique::MaxClique(AGraph agraph) : agraph_(agraph) {}

bool MaxClique::getEdge(int i, int j) const {
  if (i < 0 || i >= agraph_.numVertices()) {
    throw error("Vertex index i={} out of bounds.", i);
  }
  if (j < 0 || j >= agraph_.numVertices()) {
    throw error("Vertex index j={} out of bounds.", j);
  }
  return agraph_.getEdge(i, j);
}

std::vector<std::vector<int>>
MaxClique::getMaximumCliques(int minSize, int maxCount, const float stepLimit) {
  stepLimit_  = stepLimit;
  numVerices_ = agraph_.numVertices();

  std::vector<bool> used(numVerices_, false);
  std::vector<std::vector<int>> cliques;

  do {
    // Prepare with remaining vertices
    vertices_.clear();
    colors_.clear();
    groupColor_.clear();
    maxGroupColor_.clear();
    steps_ = 0;

    vertices_.reserve(numVerices_);
    for (int i = 0; i < agraph_.numVertices(); ++i) {
      if (!used[i]) {
        vertices_.emplace_back(i, 0);
      }
    }

    if (vertices_.size() != numVerices_) {
      throw error("Inconsistent vertices size {} vs. numVertices {}",
                  vertices_.size(),
                  numVerices_);
    }

    colors_.resize(numVerices_ + 1);
    for (auto &color : colors_) {
      color.reserve(numVerices_ + 1);
    }
    stepCount_.resize(numVerices_ + 1);

    // Main algorithm
    updateDegree(vertices_);
    sortByDegree(vertices_);
    setColorsToDegrees(vertices_);
    for (int i = 0; i < numVerices_ + 1; ++i) {
      stepCount_[i].first  = 0;
      stepCount_[i].second = 0;
    }

    // Branch-and-bound
    logging::trace("[MaxClique] branch-and-bound for {} vertices",
                   vertices_.size());
    branch(vertices_, 1);

    // Check if maximum clique in this iteration fulfills criteria
    if (maxGroupColor_.size() >= minSize) {
      cliques.push_back(maxGroupColor_);
      for (int i : maxGroupColor_) {
        used[i] = true;
        // Remove vertices already in a group
        numVerices_--;
      }
    }
  } while (maxGroupColor_.size() >= minSize && cliques.size() < maxCount &&
           numVerices_ > 0);

  return cliques;
}

void MaxClique::updateDegree(Vertices &vs) {
  for (int i = 0; i < vs.size(); i++) {
    int degree = 0;
    for (int j = 0; j < vs.size(); j++) {
      if (getEdge(vs[i].first, vs[j].first)) {
        degree++;
      }
    }
    vs[i].second = degree;
  }
}

void MaxClique::sortByDegree(Vertices &vs) {
  std::sort(vs.begin(), vs.end(), [](const Vertex &v0, const Vertex &v1) {
    return v0.second > v1.second;
  });
}

void MaxClique::setColorsToDegrees(Vertices &vs) {
  const int max_degree = vs.front().second;
  for (int i = 0; i < max_degree; i++) {
    vs[i].second = i + 1;
  }
  for (int i = max_degree; i < vs.size(); i++) {
    vs[i].second = max_degree + 1;
  }
}

void MaxClique::sortByColor(Vertices &vs) {
  int j       = 0;
  int upper_k = 1;
  int lower_k =
      static_cast<int>(maxGroupColor_.size() - groupColor_.size()) + 1;
  colors_[1].clear();
  colors_[2].clear();
  int k = 1;
  for (int i = 0; i < vs.size(); ++i) {
    int vindex = vs.at(i).first;
    k          = 1;
    while (colorCut(vindex, colors_[k])) {
      k++;
    }
    if (k > upper_k) {
      upper_k = k;
      colors_[upper_k + 1].clear();
    }
    colors_[k].push_back(vindex);
    if (k < lower_k) {
      vs.at(j).first = vindex;
      ++j;
    }
  }
  if (j > 0) {
    vs.at(j - 1).second = 0;
  }
  if (lower_k <= 0) {
    lower_k = 1;
  }
  for (k = lower_k; k <= upper_k; ++k) {
    for (int i = 0; i < colors_[k].size(); ++i) {
      vs.at(j).first  = colors_[k].at(i);
      vs.at(j).second = k;
      ++j;
    }
  }
}

bool MaxClique::colorCut(const int vindex, const ColorGroup &cg) {
  for (int i = 0; i < cg.size(); ++i)
    if (getEdge(vindex, cg.at(i)))
      return true;
  return false;
}

void MaxClique::verticesCut(const Vertices &vs0, Vertices &vs1) {
  for (int i = 0; i < vs0.size() - 1; ++i) {
    if (getEdge(vs0.back().first, vs0.at(i).first))
      vs1.emplace_back(vs0.at(i).first, 0);
  }
}

void MaxClique::branch(Vertices vs, int depth) {
  stepCount_.at(depth).first =
      (stepCount_.at(depth).first + stepCount_.at(depth - 1).first -
       stepCount_.at(depth).second);

  stepCount_.at(depth).second = (stepCount_.at(depth - 1).first);

  while (!vs.empty()) {
    if (groupColor_.size() + vs.back().second > maxGroupColor_.size()) {
      groupColor_.push_back(vs.back().first);
      Vertices vsx;
      vsx.reserve(vs.size());
      verticesCut(vs, vsx);
      if (vsx.size()) {
        ++steps_;
        if (static_cast<float>(stepCount_.at(depth).first) / steps_ <
            stepLimit_) {
          updateDegree(vsx);
          sortByDegree(vsx);
        }
        sortByColor(vsx);
        stepCount_.at(depth).first++;
        branch(vsx, depth + 1);
      } else if (groupColor_.size() > maxGroupColor_.size()) {
        maxGroupColor_ = groupColor_;
      }
      groupColor_.pop_back();
    } else {
      return;
    }
    vs.pop_back();
  }
}

} // namespace graphclique
} // namespace popart
