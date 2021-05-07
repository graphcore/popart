// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <transforms/autodiff/stitcher.hpp>

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

  return result;
}

} // namespace popart
