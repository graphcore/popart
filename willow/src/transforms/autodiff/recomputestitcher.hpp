
// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_TRANSFORMS_AUTODIFF_RECOMPUTE_STITCHER_HPP
#define GUARD_NEURALNET_TRANSFORMS_AUTODIFF_RECOMPUTE_STITCHER_HPP

#include <vector>

#include <transforms/autodiff/stitcher.hpp>

namespace popart {

/**
 * Helper class for growing gradient ops.
 *
 * Make it so that all non-gradient inputs to a backwards graph are available
 * as inputs or outputs of the forward graph by removing those inputs of the
 * backwards graph for which this is for which this currently does not hold and
 * adding fwdGraph's ops to bwdGraph to recompute those inputs inside the
 * backwards graph itself. It is exp
 **/
class RecomputeStitcher : public Stitcher {
public:
  enum class StitchIndexMode {
    /// In this mode the RecomputeStitcher will by default only stitch backward
    /// graph input indices associated with non-gradient forward graph tensors
    /// that are neither inputs nor outputs in the forward graph.
    Minimum = 0,
    /// In this mode the RecomputeStitcher will stitch only backward graph input
    /// indices associated with non-gradient forward graph tensors that are not
    /// inputs of the forward graph. Note this means outputs of the forward may
    /// be recomputed. This mode exist to preserve existing behaviour.
    AllNonInputs = 1
  };

  // Constructor.
  RecomputeStitcher(AutodiffIrInterface &dep, StitchIndexMode mode);
  virtual ~RecomputeStitcher() = default;

  /**
   * See `StitcherInterface`.
   **/
  virtual BwdGraphInfo
  stitch(const GraphId &fwdGraphId,
         const BwdGraphInfo &bwdGraphInfo,
         const nonstd::optional<std::vector<InIndex>> &stitchIndices);

  /**
   * See `Stitcher`. At the moment this will return a value in accordance to
   * the `mode` set in the constructor.
   **/
  virtual bool isDefaultStitch(const GraphId &fwdGraphId,
                               const BwdGraphInfo &bwdGraphInfo,
                               const ExpectedConnection &expInput);

  // Stitch index mode.
  StitchIndexMode mode;
};

} // namespace popart

#endif
