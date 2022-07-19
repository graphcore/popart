// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_RECOMPUTE_HPP_
#define POPART_WILLOW_INCLUDE_POPART_RECOMPUTE_HPP_

namespace popart {

enum class RecomputationType;
class Graph;

namespace recompute {
void autoAnnotate(Graph &graph, RecomputationType rctype);
void annotateRecomputeAll(Graph &graph);

} // namespace recompute
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_RECOMPUTE_HPP_
