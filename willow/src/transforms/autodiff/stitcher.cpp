// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <functional>
#include <transforms/autodiff/stitcher.hpp>
#include <popart/graph.hpp>

#include "popart/bwdgraphinfo.hpp"
#include "popart/error.hpp"
#include "popart/logging.hpp"
#include "popart/vendored/optional.hpp"
#include "transforms/autodiff/autodiffhelper.hpp"
#include "transforms/autodiff/autodiffirinterface.hpp"
#include "transforms/autodiff/stitcherinterface.hpp"

namespace popart {

Stitcher::Stitcher(AutodiffIrInterface &dep)
    : StitcherInterface{}, AutodiffHelper{dep} {}

std::vector<InIndex> Stitcher::getStitchIndices(
    const GraphId &fwdGraphId,
    const BwdGraphInfo &bwdGraphInfo,
    const nonstd::optional<std::vector<InIndex>> &stitchIndices) {

  std::vector<InIndex> result;

  if (stitchIndices) {
    result = *stitchIndices;
  } else {

    for (InIndex i = 0; i < bwdGraphInfo.expectedInputs.size(); ++i) {
      const auto &expInput = bwdGraphInfo.expectedInputs[i];
      if (isDefaultStitch(fwdGraphId, bwdGraphInfo, expInput)) {
        result.push_back(i);
      }
    }
  }

  // Check the user isn't requesting we stitch something unstitchable for this
  // stitcher.
  for (InIndex stitchIndex : result) {
    const auto &expInput = bwdGraphInfo.expectedInputs.at(stitchIndex);
    if (!isStitchable(fwdGraphId, bwdGraphInfo, expInput)) {
      auto &ir          = dep.get();
      auto &fwdGraph    = ir.getGraph(fwdGraphId);
      auto &bwdGraph    = ir.getGraph(bwdGraphInfo.bwdGraphId);
      const auto &fwdId = expInput.fwdId;
      const auto &bwdId = bwdGraph.getInputId(stitchIndex);
      throw error("[Stitcher] Unable to stitch input #{} of "
                  "{} ('{}'), which is associated with {} "
                  "'{}' of {} with this stitcher",
                  stitchIndex,
                  bwdGraph.getGraphString(),
                  bwdId,
                  expInput.type == ExpectedConnectionType::Fwd
                      ? "forward tensor"
                      : "the gradient of forward tensor",
                  fwdId,
                  fwdGraph.getGraphString());
    }
  }

  return result;
}

} // namespace popart
