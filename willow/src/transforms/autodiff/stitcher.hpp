// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_SRC_TRANSFORMS_AUTODIFF_STITCHER_HPP_
#define POPART_WILLOW_SRC_TRANSFORMS_AUTODIFF_STITCHER_HPP_

#include <transforms/autodiff/autodiffhelper.hpp>
#include <transforms/autodiff/stitcherinterface.hpp>
#include <vector>
#include <popart/bwdgraphinfo.hpp>
#include <popart/vendored/optional.hpp> // IWYU pragma: keep

#include "popart/graphid.hpp"
#include "popart/names.hpp"

namespace popart {

// Forward declarations.
class AutodiffIrInterface;

/**
 * A "stitcher" is an object that changes the IR so that all non-gradient inputs
 * to a backwards graph are guaranteed to be available as either inputs or
 * outputs of the forward graph. There are a number of strategies by which to do
 * this and this is an interface class for objects that implement these
 * strategies.
 **/
class Stitcher : public StitcherInterface, protected AutodiffHelper {
public:
  // Constructor.
  explicit Stitcher(AutodiffIrInterface &dep);
  // Destructor.
  virtual ~Stitcher() = default;

  /**
   * See `StitcherInterface`.
   **/
  virtual BwdGraphInfo
  stitch(const GraphId &fwdGraphId,
         const BwdGraphInfo &bwdGraphInfo,
         const nonstd::optional<std::vector<InIndex>> &stitchIndices) = 0;

  /**
   * Method used to compile the list of indices that are stitched when this
   * stitcher is used when `stitch` is called with `stitchIndices` unset. Used
   * to implement `getDefaultStitchIndices`.
   **/
  virtual bool isDefaultStitch(const GraphId &fwdGraphId,
                               const BwdGraphInfo &bwdGraphInfo,
                               const ExpectedConnection &expInput) = 0;

  /**
   * Method used to determine if this stitcher can stitch a specific input.
   **/
  virtual bool isStitchable(const GraphId &fwdGraphId,
                            const BwdGraphInfo &bwdGraphInfo,
                            const ExpectedConnection &expInput) = 0;

protected:
  // Helper method that returns the set of default stitch indices.
  virtual std::vector<InIndex>
  getStitchIndices(const GraphId &fwdGraphId,
                   const BwdGraphInfo &bwdGraphInfo,
                   const nonstd::optional<std::vector<InIndex>> &stitchIndices);
};

} // namespace popart

#endif // POPART_WILLOW_SRC_TRANSFORMS_AUTODIFF_STITCHER_HPP_
