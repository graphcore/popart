// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <transforms/autodiff/gradgrowermaingraph.hpp>

#include <popart/alias/aliasmodel.hpp>
#include <popart/alias/aliasmodelgrower.hpp>
#include <popart/bwdgraphinfo.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op.hpp>
#include <popart/tensorindex.hpp>
#include <popart/tensornames.hpp>

#include <transforms/autodiff/backwardsgraphcreator.hpp>
#include <transforms/autodiff/gradgrower.hpp>
#include <transforms/autodiff/gradnongradpair.hpp>
#include <transforms/autodiff/opgradregistry.hpp>
#include <transforms/autodiff/recomputestitcher.hpp>
#include <transforms/autodiff/tensorgradregistry.hpp>

namespace popart {

GradGrowerMainGraph::GradGrowerMainGraph(
    AutodiffIrInterface &dep,
    std::unique_ptr<GradGrowerOpInterface> gradOpGrower_,
    std::unique_ptr<GradGrowerLossInterface> gradLossGrower_,
    std::unique_ptr<GradGrowerSumOpInterface> gradSumOpGrower_,
    std::unique_ptr<GradGrowerGraphInterface> gradGraphGrower_)
    : GradGrowerMainGraphInterface(), AutodiffHelper(dep),
      gradOpGrower(std::move(gradOpGrower_)),
      gradLossGrower(std::move(gradLossGrower_)),
      gradSumOpGrower(std::move(gradSumOpGrower_)),
      gradGraphGrower(std::move(gradGraphGrower_)) {}

void GradGrowerMainGraph::growGradMainGraph() {

  auto &mainGraph = dep.get().getMainGraph();

  AliasModel mainGraphAliasModel;
  AliasModelGrower aliasModelGrower{mainGraphAliasModel};
  aliasModelGrower.growFullGraph(dep.get().getMainGraph(),
                                 DataDependenciesOnly::Yes);

  // We need to grow called graphs before we grow the main graph.
  FwdGraphToBwdGraphInfo calledGraphGradInfo;

  for (auto g : mainGraph.getCalledGraphs()) {

    if (calledGraphGradInfo.find(g->id) == calledGraphGradInfo.end()) {
      calledGraphGradInfo = gradGraphGrower->growBackwardsGraph(
          g->id,
          // Assume we provide all fwd outputs as gradients.
          g->getOutputIds(),
          // Require no fwd inputs as gradients as some won't be possible.
          nonstd::optional<std::vector<TensorId>>(),
          calledGraphGradInfo,
          dep.get().getSessionOptions().autodiffSettings.stitchStrategy);
    }
  }

  // definition: edge-gradient. What is output by a grad-op,
  // and which will be summed with other edge-gradients to create
  // a gradient. It is possible that an edge-gradient has the same
  // value as a gradient, if a tensor has only 1 consumer.

  // design decision w.r.t. lambda functions in this function:
  // see-sawing between lambda functions (see two following here)
  // and member functions. In general I don't like lambda functions,
  // their return types are not easily visible and capturing parameters
  // is tedious. However, I also don't like having class variables
  // which are only used in one bit of functionality, because it becomes
  // unclear whether they should be maintained in a valid state throughout
  // the objects life. In this case, I think the second is worse, so
  // going for the lambda solution.

  // Link up loss / loss scaling ops.
  Op *nonConstLossScaleOp = gradLossGrower->growLossGradients();

  // Add loss op gradients.
  auto finalLossOpId =
      dep.get().getMainGraph().getOps().find(dep.get().getFinalLossOpId());

  if (finalLossOpId == dep.get().getMainGraph().getOps().end()) {
    throw error(
        "[Autodiff] Call to growLossGradients, but finalLossOpId not found");
  }

  auto finalLossOp =
      dep.get().getMainGraph().getOp(dep.get().getFinalLossOpId());
  auto gradOps = gradOpGrower->growGradOps(finalLossOp, calledGraphGradInfo);

  logging::transform::trace("[Autodiff] Created {} gradient ops for '{}'",
                            gradOps.size(),
                            finalLossOp->str());

  std::vector<GradOpsOfNonGradOp> initGradOps = {{gradOps, finalLossOp}};

  // TODO: How will gradsProvidedFor work for subgraphs, where there is no
  //       initial op to process?
  GradGrower ad{dep};
  ad.growGrads(initGradOps,
               *gradOpGrower,
               *gradSumOpGrower,
               calledGraphGradInfo,
               mainGraphAliasModel);

  if (nonConstLossScaleOp) {
    // Only now inherit attributes for the non-const loss scaling op, if there
    // was one. The reason we do it here is because the inherit function relies
    // on the op having input or output tensors linked to it to inherit the
    // attributes from, but at the time growLossGradients is called this op's
    // outputs have yet to be grown.
    nonConstLossScaleOp->inheritPlacementAttributes(true, mainGraphAliasModel);
  }

  dep.get().setMainGraphPathFromLoss();
}

} // namespace popart
