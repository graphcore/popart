
// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_TRANSFORMS_AUTODIFF_STITCHER_FACTORY_HPP
#define GUARD_NEURALNET_TRANSFORMS_AUTODIFF_STITCHER_FACTORY_HPP

#include <memory>

#include <transforms/autodiff/stitcherinterface.hpp>
#include <popart/transforms/autodiff.hpp>

namespace popart {

// Forward declarations.
class Ir;

/**
 * A factory class for stitchers.
 **/
class StitcherFactory {
public:
  virtual ~StitcherFactory() = default;

  // Stitch a particular backwards graph.
  virtual std::unique_ptr<StitcherInterface>
  createStitcher(AutodiffIrInterface &ir,
                 AutodiffStitchStrategy stitchStrategy);
};

} // namespace popart

#endif
