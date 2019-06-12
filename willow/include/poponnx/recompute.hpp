#ifndef GUARD_NEURALNET_RECOMPUTE_HPP
#define GUARD_NEURALNET_RECOMPUTE_HPP

namespace poponnx {

enum class RecomputationType;
class Graph;

namespace recompute {
void autoAnnotate(Graph &graph, RecomputationType rctype);

} // namespace recompute
} // namespace poponnx

#endif
