// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <transforms/autodiff/safeaddfwdoutputstitcher.hpp>

#include <list>

#include <popart/graph.hpp>
#include <popart/logging.hpp>
#include <popart/op/call.hpp>

#include <transforms/autodiff/backwardsgraphcreatorhelper.hpp>

namespace popart {

SafeAddFwdOutputStitcher::SafeAddFwdOutputStitcher(AutodiffIrInterface &dep)
    : Stitcher{dep}, addFwdOutputStitcher{dep},
      recomputeStitcher{dep, RecomputeStitcher::StitchIndexMode::Minimal} {}

BwdGraphInfo SafeAddFwdOutputStitcher::stitch(
    const GraphId &fwdGraphId,
    const BwdGraphInfo &bwdGraphInfo,
    const nonstd::optional<std::vector<InIndex>> &optStitchIndices) {

  std::vector<InIndex> stitchIndices =
      getStitchIndices(fwdGraphId, bwdGraphInfo, optStitchIndices);

  // Split the indices into those to be stitched by the AddFwdOutputStitcher
  // and those to be stitched by the RecomputeStitcher (never give
  // AddFwdOutputStitcher an input it can't stitch).
  std::vector<InIndex> addFwdStitchIndices;
  std::vector<InIndex> recomputeStitchIndices;

  for (InIndex stitchIndex : stitchIndices) {
    const auto &expInput = bwdGraphInfo.expectedInputs.at(stitchIndex);
    if (addFwdOutputStitcher.isStitchable(fwdGraphId, bwdGraphInfo, expInput)) {
      addFwdStitchIndices.push_back(stitchIndex);
    } else {
      recomputeStitchIndices.push_back(stitchIndex);
    }
  }

  auto result = addFwdOutputStitcher.stitch(
      fwdGraphId, bwdGraphInfo, addFwdStitchIndices);

  // NOTE: We are implicitly making the assumption that it is safe to use the
  // indices in recomputeStitchIndices because AddFwdOutputStitcher doesn't
  // change the backward graph's inputs, and hence indices are unchange.
  // We check that assumption here in the interest of defensive programming.
  if (bwdGraphInfo.expectedInputs != result.expectedInputs) {
    throw internal_error("[SafeAddFwdOutputStitcher] AddFwdOutputStitcher has "
                         "unexpectedly changed the backwards graph inputs "
                         "(was: {}, now: {})",
                         bwdGraphInfo.expectedInputs,
                         result.expectedInputs);
  }

  result = recomputeStitcher.stitch(fwdGraphId, result, recomputeStitchIndices);

  return result;
}

bool SafeAddFwdOutputStitcher::isDefaultStitch(
    const GraphId &fwdGraphId,
    const BwdGraphInfo &bwdGraphInfo,
    const ExpectedConnection &expInput) {

  auto &ir       = dep.get();
  auto &fwdGraph = ir.getGraph(fwdGraphId);

  const auto &fwdId = expInput.fwdId;
  const auto &type  = expInput.type;

  if (type != ExpectedConnectionType::Fwd) {
    // We can only stitch non-gradient inputs.
    return false;
  }

  auto isFwdInput  = fwdGraph.hasInputId(fwdId);
  auto isFwdOutput = fwdGraph.hasOutputId(fwdId);

  if (isFwdInput || isFwdOutput) {
    // Don't stitch things that don't need stitching.
    return false;
  }

  return true;
}

bool SafeAddFwdOutputStitcher::isStitchable(
    const GraphId &fwdGraphId,
    const BwdGraphInfo &bwdGraphInfo,
    const ExpectedConnection &expInput) {

  const auto &type = expInput.type;

  if (type != ExpectedConnectionType::Fwd) {
    // We can only stitch non-gradient inputs.
    return false;
  }

  return true;
}

} // namespace popart
