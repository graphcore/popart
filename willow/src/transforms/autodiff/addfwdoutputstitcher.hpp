// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_TRANSFORMS_AUTODIFF_ADD_FWD_OUTPUT_STITCHER_HPP
#define GUARD_NEURALNET_TRANSFORMS_AUTODIFF_ADD_FWD_OUTPUT_STITCHER_HPP

#include <vector>

#include <transforms/autodiff/stitcher.hpp>

namespace popart {

/**
 * Helper class for growing gradient ops.
 *
 * Make it so that all non-gradient inputs to a backwards graph are available
 * as inputs or outputs of the forward graph by adding any required fwd tensor
 * as an output of the fwd graph and amending all call sites.
 **/
class AddFwdOutputStitcher : public Stitcher {
public:
  // Constructor.
  explicit AddFwdOutputStitcher(AutodiffIrInterface &dep);
  virtual ~AddFwdOutputStitcher() = default;

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
   * and input not an output of the forward graph but only if the forward
   * graph's call sites are limited to CallOps.
   **/
  virtual bool isDefaultStitch(const GraphId &fwdGraphId,
                               const BwdGraphInfo &bwdGraphInfo,
                               const ExpectedConnection &expInput);
};

} // namespace popart

#endif
