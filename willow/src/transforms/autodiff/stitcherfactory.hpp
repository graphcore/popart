
// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_SRC_TRANSFORMS_AUTODIFF_STITCHERFACTORY_HPP_
#define POPART_WILLOW_SRC_TRANSFORMS_AUTODIFF_STITCHERFACTORY_HPP_

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

#endif // POPART_WILLOW_SRC_TRANSFORMS_AUTODIFF_STITCHERFACTORY_HPP_
