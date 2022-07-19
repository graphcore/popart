
// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_SRC_TRANSFORMS_AUTODIFF_STITCHERINTERFACE_HPP_
#define POPART_WILLOW_SRC_TRANSFORMS_AUTODIFF_STITCHERINTERFACE_HPP_

#include <vector>
#include <popart/bwdgraphinfo.hpp>
#include <popart/vendored/optional.hpp> // IWYU pragma: keep

#include "popart/graphid.hpp"
#include "popart/names.hpp"

namespace popart {

/**
 * A "stitcher" is an object that changes the IR so that all non-gradient inputs
 * to a backwards graph are guaranteed to be available as either inputs or
 * outputs of the forward graph. There are a number of strategies by which to do
 * this and this is an interface class for objects that implement these
 * strategies.
 **/
class StitcherInterface {
public:
  // Destructor.
  virtual ~StitcherInterface() = default;

  /**
   * See Autodiff::stitch.
   **/
  virtual BwdGraphInfo
  stitch(const GraphId &fwdGraphId,
         const BwdGraphInfo &bwdGraphInfo,
         const nonstd::optional<std::vector<InIndex>> &stitchIndices) = 0;
};

} // namespace popart

#endif // POPART_WILLOW_SRC_TRANSFORMS_AUTODIFF_STITCHERINTERFACE_HPP_
