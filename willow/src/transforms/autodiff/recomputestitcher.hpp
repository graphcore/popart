
// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_TRANSFORMS_AUTODIFF_STITCHER_HPP
#define GUARD_NEURALNET_TRANSFORMS_AUTODIFF_STITCHER_HPP

#include <vector>

#include <transforms/autodiff/stitcherinterface.hpp>

namespace popart {

/**
 * Helper class for growing gradient ops.
 *
 * Make it so that all non-gradient inputs to a backwards graph are available
 * as inputs or outputs of the forward graph by removing those inputs of the
 * backwards graph for which this for which this currently does not hold and
 * adding fwdGraph's ops to bwdGraph to recompute those tensors.
 **/
class RecomputeStitcher : public StitcherInterface, private AutodiffHelper {
public:
  // Constructor.
  explicit RecomputeStitcher(AutodiffIrInterface &dep);
  virtual ~RecomputeStitcher() = default;
  // Stitch a particular backwards graph.
  virtual BwdGraphInfo
  stitch(const GraphId &fwdGraphId,
         const BwdGraphInfo &bwdGraphInfo,
         const nonstd::optional<std::vector<InIndex>> &stitchIndices);
};

} // namespace popart

#endif
