
// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_GRAD_SUM_GROWER_LOSS_HPP
#define GUARD_NEURALNET_GRAD_SUM_GROWER_LOSS_HPP

#include <popart/names.hpp>

#include <transforms/autodiff/autodiffirinterface.hpp>
#include <transforms/autodiff/gradgrower.hpp>

namespace popart {

// Forward declarations.
class Ir;
class Op;

/**
 * Interface for GradGrowerLoss.
 */
class GradGrowerLossInterface {
public:
  virtual Op *growLossGradients() = 0;
};

/**
 * Helper class for growing gradients for the loss.
 *
 * The AutodiffIrInterface dependency is passed by reference to the constructor.
 * It is the caller's responsibility to ensure the lifetime of this dependency
 * exceeds that of the GradGrowerLoss instance.
 */
class GradGrowerLoss : public GradGrowerLossInterface, private GradGrower {
public:
  // Constructor.
  explicit GradGrowerLoss(AutodiffIrInterface &dep);

  virtual Op *growLossGradients();
};

} // namespace popart

#endif
