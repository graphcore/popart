// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_LIKEOPSPATTERN_HPP
#define GUARD_NEURALNET_LIKEOPSPATTERN_HPP

#include <popart/graph.hpp>
#include <popart/patterns/pattern.hpp>

namespace popart {

// Templated pattern that replaces a _LikeOp with the equivalent _Op
template <class L> class LikeOpsPattern : public PreAliasPattern {
public:
  // Does op at the root of the
  // pattern make a match the LikeOp L?
  bool matches(Op *op) const final { return op->isConvertibleTo<L>(); }

  // If this Pattern were to be applied at op, which
  // Tensors in the subgraph centered (rooted) on op
  // would be touched?
  std::vector<const Tensor *> touches(Op *) const final { return {}; }

  // apply the pattern,
  // changes the graph of the op
  bool apply(Op *op) const final {
    // matches must have verified the correctness before this call
    auto likeOp    = static_cast<L *>(op);
    auto outTensor = likeOp->outTensor(likeOp->getOutIndex());

    auto &graph = op->getGraph();
    Op::Settings settings(graph, likeOp->name() + "_" + getPatternName());
    auto foldedOp = likeOp->foldInputTensor(settings);
    transferBaseProperties(likeOp, foldedOp.get());

    // Remove the LikeOp
    op->disconnectAllInputs();
    op->disconnectAllOutputs();
    graph.eraseOp(op->id);

    // Connect the folded Op
    foldedOp->connectOutTensor(foldedOp->getOutIndex(), outTensor->id);
    graph.moveIntoGraph(std::move(foldedOp));

    return true;
  }
};
} // namespace popart

#endif
