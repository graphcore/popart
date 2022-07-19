
// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_SRC_TRANSFORMS_AUTODIFF_GRADGROWERMAINGRAPH_HPP_
#define POPART_WILLOW_SRC_TRANSFORMS_AUTODIFF_GRADGROWERMAINGRAPH_HPP_

#include <memory>
#include <transforms/autodiff/gradgrowergraph.hpp>
#include <transforms/autodiff/gradgrowerloss.hpp>
#include <transforms/autodiff/gradgrowerop.hpp>
#include <transforms/autodiff/gradgrowersumop.hpp>

#include "transforms/autodiff/autodiffhelper.hpp"

namespace popart {
class AutodiffIrInterface;

/**
 * Interface for GradGrowerMainGraph.
 */
class GradGrowerMainGraphInterface {
public:
  // Grow backwards pass for main graph.
  virtual void growGradMainGraph() = 0;
};

/**
 * Helper class for growing gradient ops.
 *
 * Dependencies are passed by reference to the constructor. It is the caller's
 * responsibility to ensure the dependencies' lifetime exceeds that of the
 * GradGrowerMainGraph instance.
 */
class GradGrowerMainGraph : public GradGrowerMainGraphInterface,
                            private AutodiffHelper {
public:
  // Constructor.
  GradGrowerMainGraph(
      AutodiffIrInterface &dep,
      std::unique_ptr<GradGrowerOpInterface> gradOpGrower,
      std::unique_ptr<GradGrowerLossInterface> gradLossGrower,
      std::unique_ptr<GradGrowerSumOpInterface> gradSumOpGrower,
      std::unique_ptr<GradGrowerGraphInterface> gradGraphGrower);

  // Grow backwards pass.
  virtual void growGradMainGraph() override;

private:
  // Helper class to grow grad ops.
  std::unique_ptr<GradGrowerOpInterface> gradOpGrower;
  // Helper class to grow grad loss.
  std::unique_ptr<GradGrowerLossInterface> gradLossGrower;
  // Helper class to grow grad sum ops.
  std::unique_ptr<GradGrowerSumOpInterface> gradSumOpGrower;
  // Helper class to grow graphs.
  std::unique_ptr<GradGrowerGraphInterface> gradGraphGrower;
};

} // namespace popart

#endif // POPART_WILLOW_SRC_TRANSFORMS_AUTODIFF_GRADGROWERMAINGRAPH_HPP_
