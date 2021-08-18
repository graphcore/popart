// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <transforms/autodiff/addfwdoutputstitcher.hpp>

#include <list>

#include <popart/graph.hpp>
#include <popart/logging.hpp>
#include <popart/op/call.hpp>

#include <transforms/autodiff/backwardsgraphcreatorhelper.hpp>

namespace popart {

AddFwdOutputStitcher::AddFwdOutputStitcher(AutodiffIrInterface &dep)
    : Stitcher{dep} {}

BwdGraphInfo AddFwdOutputStitcher::stitch(
    const GraphId &fwdGraphId,
    const BwdGraphInfo &bwdGraphInfo,
    const nonstd::optional<std::vector<InIndex>> &optStitchIndices) {

  std::vector<InIndex> stitchIndices =
      getStitchIndices(fwdGraphId, bwdGraphInfo, optStitchIndices);

  auto &ir       = dep.get();
  auto &fwdGraph = ir.getGraph(fwdGraphId);
  auto &bwdGraph = ir.getGraph(bwdGraphInfo.bwdGraphId);

  // Remove bwdGraph inputs that we're going to recompute.
  auto bwdGraphInputs = bwdGraph.getInputIds();
  for (InIndex i : stitchIndices) {

    const auto &expInput = bwdGraphInfo.expectedInputs[i];
    const auto &fwdId    = expInput.fwdId;
    const auto &bwdId    = bwdGraphInputs.at(i);
    const auto &type     = expInput.type;

    switch (type) {
    case ExpectedConnectionType::Fwd: {

      // Do some logging.
      logging::transform::trace("[AddFwdOutputStitcher] Making '{}' an "
                                "output of {} because input #{} of {} ('{}') "
                                "is associated with '{}' and it is not "
                                "already an input of {}",
                                fwdId,
                                fwdGraph.getGraphString(),
                                i,
                                bwdGraph.getGraphString(),
                                bwdId,
                                fwdId,
                                fwdGraph.getGraphString());

      // Add as output on fwd graph.
      fwdGraph.markAsOutput(fwdId);
      OutIndex subgraphOutIndex = fwdGraph.getOutputIndex(fwdId);

      // Update all ops that call fwdGraph.
      for (Op *op : fwdGraph.getCallSiteOps()) {

        // Unable to stitch non-callop call sites at present. For LoopOps
        // the reason we can't stitch it's subgraph is that we don't currently
        // support autodiffing LoopOps and hence don't know what it's grad op
        // and associated backwards graph would look like. For IfOp the
        // limitation is more fundamental in that we are unable to expose an
        // output unless we have it available in both branches, which would
        // require an isomorphic check -- which would likely not find a match
        // for many IfOps.
        if (dynamic_cast<CallOp *>(op) != nullptr) {
          SubgraphIndex cg = 0;
          OutIndex opOutIndex =
              op->subgraphOutToOpOutIndex(cg, subgraphOutIndex);
          auto tmpId = ir.createIntermediateTensorId({"autodiff_output"});
          op->createAndConnectOutTensor(opOutIndex, tmpId);
          op->setup();
        } else {
          throw error("[AddFwdOutputStitcher] [RecomputeStitcher] "
                      "Unsupported connection type ({})",
                      static_cast<int>(type));
        }
      }

      break;
    }
    case ExpectedConnectionType::FwdGrad: {

      // Error because not a non-gradient forward tensor.
      throw error("[AddFwdOutputStitcher] Unable to stitch input #{} of "
                  "{} ('{}') because it is associated with the gradient "
                  "tensor '{}' of {}",
                  i,
                  bwdGraph.getGraphString(),
                  bwdId,
                  fwdId,
                  fwdGraph.getGraphString());
    }
    default: {

      throw internal_error("[AddFwdOutputStitcher] Unsupported connection type "
                           "({})",
                           static_cast<int>(type));
    }
    }
  }

  // Backwards graph is unchanged.
  return bwdGraphInfo;
}

bool AddFwdOutputStitcher::isDefaultStitch(const GraphId &fwdGraphId,
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

  for (Op *op : fwdGraph.getCallSiteOps()) {
    if (dynamic_cast<CallOp *>(op) == nullptr) {
      // We can't deal with non-callop call-sites at present.
      return false;
    }
  }

  return true;
}

bool AddFwdOutputStitcher::isStitchable(const GraphId &fwdGraphId,
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

  auto isFwdOutput = fwdGraph.hasOutputId(fwdId);

  if (isFwdOutput) {
    // We can't add tensors that are already outputs as outputs.
    return false;
  }

  for (Op *op : fwdGraph.getCallSiteOps()) {
    if (dynamic_cast<CallOp *>(op) == nullptr) {
      // We can't deal with non-callop call-sites at present.
      return false;
    }
  }

  return true;
}

} // namespace popart
