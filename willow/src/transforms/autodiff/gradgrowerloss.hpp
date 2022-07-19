
// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_SRC_TRANSFORMS_AUTODIFF_GRADGROWERLOSS_HPP_
#define POPART_WILLOW_SRC_TRANSFORMS_AUTODIFF_GRADGROWERLOSS_HPP_

#include <transforms/autodiff/autodiffhelper.hpp>

namespace popart {

// Forward declarations.
class Op;
class AutodiffIrInterface;

/**
 * Interface for GradGrowerLoss.
 */
class GradGrowerLossInterface {
public:
  virtual ~GradGrowerLossInterface() = default;
  virtual Op *growLossGradients()    = 0;
};

/**
 * Helper class for growing gradients for the loss.
 *
 * The AutodiffIrInterface dependency is passed by reference to the constructor.
 * It is the caller's responsibility to ensure the lifetime of this dependency
 * exceeds that of the GradGrowerLoss instance.
 */
class GradGrowerLoss : public GradGrowerLossInterface, private AutodiffHelper {
public:
  // Constructor.
  explicit GradGrowerLoss(AutodiffIrInterface &dep);
  virtual ~GradGrowerLoss() = default;

  virtual Op *growLossGradients() override;
};

} // namespace popart

#endif // POPART_WILLOW_SRC_TRANSFORMS_AUTODIFF_GRADGROWERLOSS_HPP_
