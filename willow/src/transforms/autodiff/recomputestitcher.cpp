// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <transforms/autodiff/recomputestitcher.hpp>

#include <list>

#include <popart/graph.hpp>
#include <popart/logging.hpp>

#include <transforms/autodiff/backwardsgraphcreatorhelper.hpp>

namespace popart {

RecomputeStitcher::RecomputeStitcher(AutodiffIrInterface &dep,
                                     StitchIndexMode mode_)
    : Stitcher{dep}, mode{mode_} {}

BwdGraphInfo RecomputeStitcher::stitch(
    const GraphId &fwdGraphId,
    const BwdGraphInfo &bwdGraphInfo,
    const nonstd::optional<std::vector<InIndex>> &optStitchIndices) {

  std::vector<InIndex> stitchIndices =
      getStitchIndices(fwdGraphId, bwdGraphInfo, optStitchIndices);

  auto &ir       = dep.get();
  auto &fwdGraph = ir.getGraph(fwdGraphId);
  auto &bwdGraph = ir.getGraph(bwdGraphInfo.bwdGraphId);

  // Derive a list of inputs that must not be pruned. We have no business
  // removing inputs that aren't in the stitch parameter list. We need to
  // compute this ahead of time, because some inputs get removed before
  // we get to pruning.
  std::vector<InIndex> protectedInputIndices;
  const InIndex numIns = bwdGraphInfo.expectedInputs.size();
  for (InIndex index = 0; index < numIns; ++index) {
    if (std::find(stitchIndices.begin(), stitchIndices.end(), index) ==
        stitchIndices.end()) {
      protectedInputIndices.push_back(index);
    }
  }

  // Remove bwdGraph inputs that we're going to recompute.
  auto bwdGraphInputs = bwdGraph.getInputIds();
  for (InIndex i : stitchIndices) {

    const auto &expInput = bwdGraphInfo.expectedInputs[i];
    const auto &fwdId    = expInput.fwdId;
    const auto &bwdId    = bwdGraphInputs.at(i);
    const auto &type     = expInput.type;

    // NOTE: We could choose to not recompute graph inputs that are associated
    // with outputs of fwdGraph (as this is supported by grad op creation code)
    // but we don't do this here to preserve existing behaviour.

    switch (type) {
    case ExpectedConnectionType::Fwd: {

      if (!fwdGraph.hasInputId(fwdId)) {
        logging::transform::trace(
            "[RecomputeStitcher] Recomputing input #{} of "
            "{} ('{}'), which is associated with tensor "
            "'{}' of {}",
            i,
            bwdGraph.getGraphString(),
            bwdId,
            fwdId,
            fwdGraph.getGraphString());
        bwdGraph.removeInput(bwdId);
        // Any protected indices following the input we just removed are now
        // one index earlier.
        std::transform(protectedInputIndices.begin(),
                       protectedInputIndices.end(),
                       protectedInputIndices.begin(),
                       [i](InIndex in) { return (in > i) ? (in - 1) : in; });
      } else {
        throw error("[RecomputeStitcher] Unable to stitch input #{} of "
                    "{} ('{}') because it is associated with forward tensor "
                    "'{}', which is an input of {}",
                    i,
                    bwdGraph.getGraphString(),
                    bwdId,
                    fwdId,
                    fwdGraph.getGraphString());
      }
      break;
    }
    case ExpectedConnectionType::FwdGrad: {

      throw error("[RecomputeStitcher] Unable to stitch input #{} of "
                  "{} ('{}') because it is associated with the gradient "
                  "tensor '{}' of {}",
                  i,
                  bwdGraph.getGraphString(),
                  bwdId,
                  fwdId,
                  fwdGraph.getGraphString());
      break;
    }
    default: {

      throw internal_error("[RecomputeStitcher] Unsupported connection type "
                           "({})",
                           static_cast<int>(type));
    }
    }
  }

  // Copy everything from fwd to bwd.
  bwdGraph.copyFrom(
      fwdGraph, Graph::CopyInputMarkings::Yes, Graph::CopyOutputMarkings::No);

  BackwardsGraphCreatorHelper::doPrune(
      bwdGraph,
      protectedInputIndices,
      BackwardsGraphCreatorHelper::WarnIfProtectedInputCouldHaveBeenRemoved::
          No);

  BackwardsGraphCreatorHelper helper{fwdGraph, bwdGraph};
  return helper.makeGradInfo();
}

bool RecomputeStitcher::isDefaultStitch(const GraphId &fwdGraphId,
                                        const BwdGraphInfo &bwdGraphInfo,
                                        const ExpectedConnection &expInput) {

  const auto &fwdId = expInput.fwdId;
  const auto &type  = expInput.type;

  if (type != ExpectedConnectionType::Fwd) {
    // We can only stitch non-gradient inputs.
    return false;
  }

  auto &fwdGraph = dep.get().getGraph(fwdGraphId);

  auto isFwdInput  = fwdGraph.hasInputId(fwdId);
  auto isFwdOutput = fwdGraph.hasOutputId(fwdId);

  switch (mode) {
  case StitchIndexMode::Minimal: {
    return !isFwdInput && !isFwdOutput;
  }
  case StitchIndexMode::AllNonInputs: {
    return !isFwdInput;
  }
  default: {
    throw internal_error("[RecomputeStitcher] Unsupported mode ({})",
                         static_cast<int>(mode));
  }
  }
}

bool RecomputeStitcher::isStitchable(const GraphId &fwdGraphId,
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
