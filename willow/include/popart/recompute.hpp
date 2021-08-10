// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_RECOMPUTE_HPP
#define GUARD_NEURALNET_RECOMPUTE_HPP

namespace popart {

enum class RecomputationType;
class Graph;

namespace recompute {
void autoAnnotate(Graph &graph, RecomputationType rctype);
void annotateRecomputeAll(Graph &graph);

} // namespace recompute
} // namespace popart

#endif
