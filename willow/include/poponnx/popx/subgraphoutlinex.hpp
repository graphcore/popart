#ifndef GUARD_NEURALNET_SUBGRAPHOUTLINEX_HPP
#define GUARD_NEURALNET_SUBGRAPHOUTLINEX_HPP

#include <vector>
#include <poponnx/op.hpp>

#include <poponnx/subgraph/subgraph.hpp>

namespace poponnx {

class SubgraphOutlinex {

  std::vector<std::unique_ptr<Op>> subgraphOps;

  int64_t getNextSubgraphId();

  bool canApplyMatch(const std::vector<Op *> &ops, fwtools::subgraph::Match &m);

public:
  SubgraphOutlinex();

  std::vector<Op *> getOutlineView(const std::vector<Op *> &schedule,
                                   const Ir &ir);
};

} // namespace poponnx

#endif // GUARD_NEURALNET_SUBGRAPHOUTLINEX_HPP
