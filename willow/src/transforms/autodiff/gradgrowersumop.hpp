
// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_SRC_TRANSFORMS_AUTODIFF_GRADGROWERSUMOP_HPP_
#define POPART_WILLOW_SRC_TRANSFORMS_AUTODIFF_GRADGROWERSUMOP_HPP_

#include <string>
#include <transforms/autodiff/autodiffhelper.hpp>
#include <vector>

namespace popart {

// Forward declarations.
class Graph;
class Op;
class Tensor;
class AliasModel;
class AutodiffIrInterface;

/**
 * Interface for GradGrowerSumOp.
 */
class GradGrowerSumOpInterface {
public:
  virtual ~GradGrowerSumOpInterface() = default;
  // Grow a grad sum op, combining a number of gradients into one.
  virtual Op *growGradSumOp(Graph &bwdGraph,
                            Tensor *target,
                            const std::vector<Tensor *> &toSum,
                            AliasModel &mainGraphAliasModel) = 0;
};

/**
 * Helper class for growing gradient ops. It is up to the user to ensure that
 * GradGrowerSumOp does not outlive it's dependencies.
 *
 * The AutodiffIrInterface dependency is passed by reference to the constructor.
 * It is the caller's responsibility to ensure the lifetime of this dependency
 * exceeds that of the GradGrowerSumOp instance.
 */
class GradGrowerSumOp : public GradGrowerSumOpInterface,
                        private AutodiffHelper {
public:
  // Constructor.
  explicit GradGrowerSumOp(AutodiffIrInterface &dep);
  virtual ~GradGrowerSumOp() = default;

  // Grow a grad sum op, combining a number of gradients into one.
  virtual Op *growGradSumOp(Graph &bwdGraph,
                            Tensor *target,
                            const std::vector<Tensor *> &toSum,
                            AliasModel &bwdGraphAliasModel) override;

  // Prefix for grad sum operations.
  static std::string getGradSumOpNamePrefix();
};

} // namespace popart

#endif // POPART_WILLOW_SRC_TRANSFORMS_AUTODIFF_GRADGROWERSUMOP_HPP_
