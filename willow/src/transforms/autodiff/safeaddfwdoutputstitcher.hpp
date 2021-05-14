// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_TRANSFORMS_AUTODIFF_SAFE_ADD_FWD_OUTPUT_STITCHER_HPP
#define GUARD_NEURALNET_TRANSFORMS_AUTODIFF_SAFE_ADD_FWD_OUTPUT_STITCHER_HPP

#include <vector>

#include <transforms/autodiff/addfwdoutputstitcher.hpp>
#include <transforms/autodiff/recomputestitcher.hpp>
#include <transforms/autodiff/stitcher.hpp>

namespace popart {

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

#endif
