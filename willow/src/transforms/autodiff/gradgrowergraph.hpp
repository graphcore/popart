// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_GRAD_GROWER_GRAPH_HPP
#define GUARD_NEURALNET_GRAD_GROWER_GRAPH_HPP

#include <functional>
#include <memory>

#include <popart/names.hpp>

#include <popart/transforms/autodiff.hpp>

#include <transforms/autodiff/autodiffhelper.hpp>

namespace popart {

// Forward declaration.
class StitcherFactory;

/**
 * Interface for GradGrowerSubgraph.
 */
class GradGrowerGraphInterface {
public:
  // Shorthand.
  using TensorIds = std::vector<TensorId>;

  // Grow backwards pass for subgraph and all descendants.
  virtual FwdGraphToBwdGraphInfo
  growBackwardsGraph(const GraphId &fwdGraphId,
                     const TensorIds &gradsProvidedForTensors,
                     const nonstd::optional<TensorIds> &gradsRequiredForTensors,
                     const FwdGraphToBwdGraphInfo &calledGraphResults,
                     Autodiff::StitchStrategy stitchStrategy) = 0;

  virtual ~GradGrowerGraphInterface() {}
};

/**
 * Helper class for recursively growing backwards subgraphs.
 */
class GradGrowerGraph : public GradGrowerGraphInterface,
                        private AutodiffHelper {
public:
  // Constructor.
  GradGrowerGraph(AutodiffIrInterface &dep);

  // Grow backwards pass.
  virtual FwdGraphToBwdGraphInfo
  growBackwardsGraph(const GraphId &fwdGraphId,
                     const TensorIds &gradsProvidedForTensors,
                     const nonstd::optional<TensorIds> &gradsRequiredForTensors,
                     const FwdGraphToBwdGraphInfo &calledGraphResults,
                     Autodiff::StitchStrategy stitchStrategy) override;

private:
  // Helper class to create stitchers (add a setter if it helps with testing).
  std::unique_ptr<StitcherFactory> stitcherFactory;
};

} // namespace popart

#endif
