// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_TRANSFORMS_AUTODIFF_ADD_FWD_OUTPUT_STITCHER_HPP
#define GUARD_NEURALNET_TRANSFORMS_AUTODIFF_ADD_FWD_OUTPUT_STITCHER_HPP

#include <transforms/autodiff/stitcher.hpp>
#include <vector>

#include "popart/bwdgraphinfo.hpp"
#include "popart/graphid.hpp"
#include "popart/names.hpp"

namespace nonstd {
namespace optional_lite {
template <typename T> class optional;
} // namespace optional_lite
} // namespace nonstd

namespace popart {
class AutodiffIrInterface;

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
   * and input nor an output of the forward graph but only if the forward
   * graph's call sites are limited to CallOps.
   **/
  virtual bool isDefaultStitch(const GraphId &fwdGraphId,
                               const BwdGraphInfo &bwdGraphInfo,
                               const ExpectedConnection &expInput);

  /**
   * See `Stitcher`. We can stitch any non-gradient input that is is not already
   * an output.
   **/
  virtual bool isStitchable(const GraphId &fwdGraphId,
                            const BwdGraphInfo &bwdGraphInfo,
                            const ExpectedConnection &expInput);
};

} // namespace popart

#endif
