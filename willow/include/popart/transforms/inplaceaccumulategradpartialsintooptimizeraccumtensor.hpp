// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_TRANSFORMS_INPLACEACCUMULATEGRADPARTIALSINTOOPTIMIZERACCUMTENSOR_HPP_
#define POPART_WILLOW_INCLUDE_POPART_TRANSFORMS_INPLACEACCUMULATEGRADPARTIALSINTOOPTIMIZERACCUMTENSOR_HPP_

#include <cstddef>
#include <string>
#include <popart/transforms/transform.hpp>

namespace popart {

class Graph;

/**
 * \brief Replaces an accumulation tree consumed by an AccumulateOp (which has
 * its own accumulator tensor), with an accumulation tree directly on the
 * AccumulateOp's accumulator tensor, thereby removing one allocation from the
 * graph (the accumulation tree's original accumulation tensor).
 *
 *
 * More precisely:
 *
 * Init
 *  |
 * dW0              pW0
 *   \             /
 *   AddLhsInPlace0
 *         |
 *        dW1              pW1
 *           \             /
 *           AddLhsInPlace1
 *                 |              A
 *                dw2   accum ----|
 *                   \    |
 *                   Accumulate3
 *                        |
 *                      accum'
 *                        |
 *                        B
 *
 * Becomes:
 *
 *  A
 *  |
 * accum       pW0
 *   \         /
 *   Accumulate
 *       |
 *       dW1         pW1
 *         \         /
 *         Accumulate
 *             |
 *           accum'
 *             |
 *             B
 *
 * See below comment for more discussion of the conditions required to be able
 * to perform this transform.
 *
 * The primary use case of this is a decomposed grad sum whose addition tree is
 * fed into an AccumulateOp as part of the optimiser step.
 */
class InplaceAccumulateGradPartialsIntoOptimizerAccumTensor final
    : public Transform {
public:
  static std::size_t id();

  InplaceAccumulateGradPartialsIntoOptimizerAccumTensor();
  ~InplaceAccumulateGradPartialsIntoOptimizerAccumTensor() final;

  bool apply(Graph &graph) const final;

  std::size_t getId() const final { return id(); }

  std::string getName() const final {
    return "InplaceAccumulateGradPartialsIntoOptimizerAccumTensor";
  }
};

} // namespace popart

/*
  Under what conditions can we perform this transform?

    1. The decomposition is mathematically correct.
    2. Tree accum must be produced by an InitOp with InitType::Zero. It cannot
       be a graph input.
    3. Tree accum, and none of its updated incarnations throughout the addition
       tree, can have ANY consumers other than the next op in the tree.
    4. Optimiser accum can be a graph input or ouput (or neither).

  Explanation:

  1.

  It is only mathematically valid to decompose AccumulateOps whose
  AccumulationType distributes over addition (Add, DampenedAdd,
  DampenedAddSquare).

  -----

  2, 3.

  For there to be any point in doing this optimisation, we must know that the
  tree accum represents a new allocation, otherwise there is no extra allocation
  to optimise away. Consider when the tree accum is the output of some multiply
  op, that allocation will still be there even if we disconnect the tree accum
  from this tree. If, on the other hand, the tree accum is produced by an
  InitOp, then it is a new allocation, and replacing it with the optimiser accum
  will remove an allocation from the graph. Note the InitOp must have
  InitType::Zero for this to be numerically correct.

  What else in the Ir represents a new allocation?

  Graph inputs:
  We could be in a subgraph and the tree accum a graph input. If the graph input
  is a new allocation (it gets copied, not aliased), there is something to
  optimise away. However, in what cases is there a subgraph? They can be created
  by outlining; however this transform happens before outlining. There can also
  be user defined subgraphs, through if ops for example. Since the primary goal
  of this transform is to target gradient trees going into optimiser steps, for
  simplicity's sake [^1] we make no attempt to optimise this case.

  Thus, we only consider the case where the tree accum is produced by an InitOp
  with InitType::Zero. The transform will have to remove the InitOp (as well as
  the tree accum) when replacing with the optimiser accum.

  [^1] This would be complex as we would have to check, for every callsite of
  this subgraph, is the parent tensor zero, and does the parent tensor have
  other consumers. We would have to know that the copy into the subgraph is not
  going to be optimised away by AZC, making this no longer a new allocation.

  -----

  4.

  We have discussed the conditions on the initial tree accum, but what about the
  optimiser accum?

  As we are not removing it, only adding more inplace operations to the existing
  inplace operation on it, it does not matter whether it is a graph input.
  Similarly, it does not matter if the updated optimiser accum tensor is a graph
  output. Even if the original optimiser accum is an input and the updated accum
  an output, we can still apply this transform; we are just performing more
  AccumulateOps inbetween.
*/

#endif // POPART_WILLOW_INCLUDE_POPART_TRANSFORMS_INPLACEACCUMULATEGRADPARTIALSINTOOPTIMIZERACCUMTENSOR_HPP_
