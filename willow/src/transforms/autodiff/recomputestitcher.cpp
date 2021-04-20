// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <transforms/autodiff/recomputestitcher.hpp>

#include <list>

#include <popart/graph.hpp>
#include <popart/logging.hpp>

#include <transforms/autodiff/backwardsgraphcreatorhelper.hpp>
//#include <popart/op.hpp>
//#include <popart/tensorindex.hpp>

namespace popart {

RecomputeStitcher::RecomputeStitcher(AutodiffIrInterface &dep)
    : StitcherInterface{}, AutodiffHelper{dep} {}

BwdGraphInfo RecomputeStitcher::stitch(
    const GraphId &fwdGraphId,
    const BwdGraphInfo &bwdGraphInfo,
    const nonstd::optional<std::vector<InIndex>> &stitchIndices) {
  // Use given bwdGraphInfo as a starting point, but recalc expectedInputs.
  BwdGraphInfo result   = bwdGraphInfo;
  result.expectedInputs = {};

  auto &ir = dep.get();

  auto &fwdGraph = ir.getGraph(fwdGraphId);
  auto &bwdGraph = ir.getGraph(bwdGraphInfo.bwdGraphId);

  // Remove bwdGraph inputs that we're going to recompute.
  auto bwdGraphInputs = bwdGraph.getInputIds();
  for (InIndex i = 0; i < bwdGraphInfo.expectedInputs.size(); ++i) {

    if (stitchIndices) {
      // Avoid stitching things that are not listed in a provided stitchIndices.
      auto it = std::find(stitchIndices->begin(), stitchIndices->end(), i);
      if (it == stitchIndices->end()) {
        continue;
      }
    }

    const auto &expInput = bwdGraphInfo.expectedInputs[i];
    const auto &fwdId    = expInput.fwdId;
    const auto &bwdId    = bwdGraphInputs.at(i);
    const auto &type     = expInput.type;

    // NOTE: We could choose to not recompute graph inputs that are associated
    // with outputs of fwdGraph (as this is supported by grad op creation code)
    // but we don't do this here to preserve existing behaviour.

    switch (type) {
    case ExpectedConnectionType::Fwd: {
      if (fwdGraph.hasInputId(fwdId)) {
        logging::transform::trace("[Autodiff] Keeping input #{} of {} ('{}') "
                                  "which is associated with '{}', an input "
                                  "of {}",
                                  i,
                                  bwdGraph.getGraphString(),
                                  bwdId,
                                  fwdId,
                                  fwdGraph.getGraphString());
        result.expectedInputs.push_back(expInput);
      } else {
        logging::transform::trace("[Autodiff] Removing input #{} of {} "
                                  "('{}') because it's associated with '{}' "
                                  "which is not an input of {}",
                                  i,
                                  bwdGraph.getGraphString(),
                                  bwdId,
                                  fwdId,
                                  fwdGraph.getGraphString());
        bwdGraph.removeInput(bwdId);
      }
      break;
    }
    case ExpectedConnectionType::FwdGrad:
    default: {
      logging::transform::trace("[Autodiff] Keeping input #{} of {} ('{}') "
                                "which is associated with the gradient of "
                                "tensor '{}' of {}",
                                i,
                                bwdGraph.getGraphString(),
                                bwdId,
                                fwdId,
                                fwdGraph.getGraphString());
      result.expectedInputs.push_back(expInput);
      break;
    }
    }
  }

  // Copy everything from fwd to bwd.
  bwdGraph.copyFrom(
      fwdGraph, Graph::CopyInputMarkings::Yes, Graph::CopyOutputMarkings::No);
  BackwardsGraphCreatorHelper::doPrune(bwdGraph);

  BackwardsGraphCreatorHelper helper{fwdGraph, bwdGraph};
  return helper.makeGradInfo();
}

} // namespace popart
