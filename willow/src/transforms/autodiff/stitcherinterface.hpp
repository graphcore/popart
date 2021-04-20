
// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_TRANSFORMS_AUTODIFF_STITCHER_INTERFACE_HPP
#define GUARD_NEURALNET_TRANSFORMS_AUTODIFF_STITCHER_INTERFACE_HPP

#include <popart/bwdgraphinfo.hpp>

#include <transforms/autodiff/autodiffhelper.hpp>
#include <transforms/autodiff/autodiffirinterface.hpp>

#include <popart/vendored/optional.hpp>

namespace popart {

// Forward declarations.
class Op;

/**
 * A "stitcher" is an object that changes the IR so that all non-gradient inputs
 * to a backwards graph are guaranteed to be available as either inputs or
 * outputs of the forward graph. This abstract is the base class for such
 * stitcher objects.
 **/
class StitcherInterface {
public:
  virtual ~StitcherInterface() = default;
  // Stitch a particular backwards graph.
  virtual BwdGraphInfo
  stitch(const GraphId &fwdGraphId,
         const BwdGraphInfo &bwdGraphInfo,
         const nonstd::optional<std::vector<InIndex>> &stitchIndices) = 0;
};

} // namespace popart

#endif
