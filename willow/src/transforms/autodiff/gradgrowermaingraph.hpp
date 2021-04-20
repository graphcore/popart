
// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_GRAD_GROWER_MAIN_GRAPH_HPP
#define GUARD_NEURALNET_GRAD_GROWER_MAIN_GRAPH_HPP

#include <functional>
#include <memory>

#include <popart/names.hpp>

#include <transforms/autodiff/autodiffirinterface.hpp>
#include <transforms/autodiff/gradgrowergraph.hpp>
#include <transforms/autodiff/gradgrowerloss.hpp>
#include <transforms/autodiff/gradgrowerop.hpp>
#include <transforms/autodiff/gradgrowersumop.hpp>

namespace popart {

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
  virtual void growGradMainGraph();

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

#endif
