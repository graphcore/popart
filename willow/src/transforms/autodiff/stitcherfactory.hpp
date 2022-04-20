
// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_TRANSFORMS_AUTODIFF_STITCHER_FACTORY_HPP
#define GUARD_NEURALNET_TRANSFORMS_AUTODIFF_STITCHER_FACTORY_HPP

#include <memory>

#include "popart/sessionoptions.hpp"

namespace popart {

// Forward declarations.
class AutodiffIrInterface;
class StitcherInterface;

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
