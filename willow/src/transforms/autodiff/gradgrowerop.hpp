
// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_GRAD_GROWER_OP_HPP
#define GUARD_NEURALNET_GRAD_GROWER_OP_HPP

#include <transforms/autodiff/autodiffhelper.hpp>
#include <vector>
#include <popart/bwdgraphinfo.hpp>

namespace popart {

// Forward declarations.
class Op;
class AutodiffIrInterface;
class Graph;

/**
 * Interface for GradGrowerOp.
 */
class GradGrowerOpInterface {
public:
  virtual ~GradGrowerOpInterface() = default;
  // Grow a collection of gradient ops for a forward op.
  virtual std::vector<Op *>
  growGradOps(Graph &bwdGraph,
              Op *forwardOp,
              const FwdGraphToBwdGraphInfo &calledGraphsGradInfo) = 0;
};

/**
 * Helper class for growing gradient ops.
 *
 * The AutodiffIrInterface dependency is passed by reference to the constructor.
 * It is the caller's responsibility to ensure the lifetime of this dependency
 * exceeds that of the GradGrowerOp instance.
 */
class GradGrowerOp : public GradGrowerOpInterface, private AutodiffHelper {
public:
  // Constructor.
  explicit GradGrowerOp(AutodiffIrInterface &dep);
  virtual ~GradGrowerOp() = default;

  // Grow a collection of gradient ops for a forward op.
  virtual std::vector<Op *>
  growGradOps(Graph &bwdGraph,
              Op *forwardOp,
              const FwdGraphToBwdGraphInfo &calledGraphsGradInfo) override;
};

} // namespace popart

#endif
