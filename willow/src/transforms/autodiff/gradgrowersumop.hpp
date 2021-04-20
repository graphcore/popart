
// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_GRAD_SUM_GROWER_OP_HPP
#define GUARD_NEURALNET_GRAD_SUM_GROWER_OP_HPP

#include <memory>

#include <popart/names.hpp>

#include <transforms/autodiff/autodiffhelper.hpp>
#include <transforms/autodiff/autodiffirinterface.hpp>

namespace popart {

// Forward declarations.
class Ir;
class Op;
class Tensor;

/**
 * Interface for GradGrowerSumOp.
 */
class GradGrowerSumOpInterface {
public:
  virtual ~GradGrowerSumOpInterface() = default;
  // Grow a grad sum op, combining a number of gradients into one.
  virtual Op *growGradSumOp(Tensor *target,
                            const std::vector<Tensor *> &toSum) = 0;
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
  virtual Op *growGradSumOp(Tensor *target, const std::vector<Tensor *> &toSum);

  // Prefix for grad sum operations.
  static std::string getGradSumOpNamePrefix();
};

} // namespace popart

#endif
