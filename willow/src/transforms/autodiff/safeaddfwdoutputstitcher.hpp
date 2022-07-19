// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_SRC_TRANSFORMS_AUTODIFF_SAFEADDFWDOUTPUTSTITCHER_HPP_
#define POPART_WILLOW_SRC_TRANSFORMS_AUTODIFF_SAFEADDFWDOUTPUTSTITCHER_HPP_

#include <transforms/autodiff/addfwdoutputstitcher.hpp>
#include <transforms/autodiff/recomputestitcher.hpp>
#include <transforms/autodiff/stitcher.hpp>
#include <vector>

#include "popart/bwdgraphinfo.hpp"
#include "popart/graphid.hpp"
#include "popart/names.hpp"
#include "popart/vendored/optional.hpp" // IWYU pragma: keep

namespace popart {
class AutodiffIrInterface;

/**
 * Helper class for growing gradient ops.
 *
 * Use AddFwdOutputStitcher where possible, but revert to RecomputeStitcher
 * where this is not possible.
 **/
class SafeAddFwdOutputStitcher : public Stitcher {
public:
  // Constructor.
  explicit SafeAddFwdOutputStitcher(AutodiffIrInterface &dep);
  virtual ~SafeAddFwdOutputStitcher() = default;

  /**
   * See `StitcherInterface`.
   **/
  virtual BwdGraphInfo
  stitch(const GraphId &fwdGraphId,
         const BwdGraphInfo &bwdGraphInfo,
         const nonstd::optional<std::vector<InIndex>> &stitchIndices);

  /**
   * See `Stitcher`. This will return `true` for any backwards graph input
   * that is associated with a non-gradient forward tensor that is neither
   * and input nor an output of the forward graph but only if the forward
   * graph's call sites are limited to CallOps.
   **/
  virtual bool isDefaultStitch(const GraphId &fwdGraphId,
                               const BwdGraphInfo &bwdGraphInfo,
                               const ExpectedConnection &expInput);

  /**
   * See `Stitcher`. We can stitch any non-gradient input.
   **/
  virtual bool isStitchable(const GraphId &fwdGraphId,
                            const BwdGraphInfo &bwdGraphInfo,
                            const ExpectedConnection &expInput);

private:
  // Stitchers used under the hood.
  AddFwdOutputStitcher addFwdOutputStitcher;
  RecomputeStitcher recomputeStitcher;
};

} // namespace popart

#endif // POPART_WILLOW_SRC_TRANSFORMS_AUTODIFF_SAFEADDFWDOUTPUTSTITCHER_HPP_
