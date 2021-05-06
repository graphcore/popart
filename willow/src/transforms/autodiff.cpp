// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <popart/transforms/autodiff.hpp>

#include <popart/bwdgraphinfo.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>

#include <transforms/autodiff/autodiffiradapter.hpp>
#include <transforms/autodiff/backwardsgraphcreator.hpp>
#include <transforms/autodiff/backwardsgraphcreatorhelper.hpp>
#include <transforms/autodiff/gradgrowergraph.hpp>
#include <transforms/autodiff/gradgrowerloss.hpp>
#include <transforms/autodiff/gradgrowermaingraph.hpp>
#include <transforms/autodiff/gradgrowerop.hpp>
#include <transforms/autodiff/gradgrowersumop.hpp>
#include <transforms/autodiff/recomputestitcher.hpp>
#include <transforms/autodiff/stitcherfactory.hpp>

namespace popart {

std::size_t Autodiff::id() { return typeid(Autodiff).hash_code(); }

Autodiff::Autodiff()
    : Transform(), stitcherFactory(std::make_unique<StitcherFactory>()) {}

Autodiff::~Autodiff() = default;

bool Autodiff::apply(Graph &graph) const { return applyToIr(graph.getIr()); }

bool Autodiff::applyToIr(Ir &ir) const {

  if (ir.isTesting()) {
    throw internal_error(
        "Call to Autodiff::apply() in Testing mode is not valid");
  }

  AutodiffIrAdapter adapter{ir};
  auto gradGrowerOp    = std::make_unique<GradGrowerOp>(adapter);
  auto gradGrowerLoss  = std::make_unique<GradGrowerLoss>(adapter);
  auto gradGrowerSumOp = std::make_unique<GradGrowerSumOp>(adapter);
  auto gradGrowerGraph = std::make_unique<GradGrowerGraph>(adapter);
  GradGrowerMainGraph gradMainGraphGrower(adapter,
                                          std::move(gradGrowerOp),
                                          std::move(gradGrowerLoss),
                                          std::move(gradGrowerSumOp),
                                          std::move(gradGrowerGraph));
  gradMainGraphGrower.growGradMainGraph();

  return true;
}

FwdGraphToBwdGraphInfo
Autodiff::apply(Ir &ir,
                const GraphId &fwdGraphId,
                const TensorIds &gradsProvidedForTensors,
                const nonstd::optional<TensorIds> &gradsRequiredForFwdId,
                const FwdGraphToBwdGraphInfo &calledGraphsGradInfo,
                StitchStrategy stitchStrategy) {

  AutodiffIrAdapter adapter{ir};
  GradGrowerGraph gradGraphGrower{adapter};

  return gradGraphGrower.growBackwardsGraph(fwdGraphId,
                                            gradsProvidedForTensors,
                                            gradsRequiredForFwdId,
                                            calledGraphsGradInfo,
                                            stitchStrategy);
}

BwdGraphInfo Autodiff::createBwdGraph(
    Ir &ir,
    const GraphId &fwdGraphId,
    const TensorIds &gradsProvidedForTensors,
    const nonstd::optional<TensorIds> &gradsRequiredForFwdId,
    const FwdGraphToBwdGraphInfo &calledGraphsGradInfo) {

  AutodiffIrAdapter adapter{ir};
  BackwardsGraphCreator bwdPassCreator{adapter};

  auto &fwdGraph     = ir.getGraph(fwdGraphId);
  GraphId bwdGraphId = bwdPassCreator.genNewBwdGraphId(fwdGraphId);

  return bwdPassCreator.createBackwardsGraph(
      fwdGraph, bwdGraphId, gradsRequiredForFwdId, calledGraphsGradInfo);
}

BwdGraphInfo
Autodiff::stitch(Ir &ir,
                 const GraphId &fwdGraphId,
                 const BwdGraphInfo &bwdGraphInfo,
                 StitchStrategy stitchStrategy,
                 const nonstd::optional<std::vector<InIndex>> &stitchIndices) {

  AutodiffIrAdapter adapter{ir};
  auto stitcher = stitcherFactory->createStitcher(adapter, stitchStrategy);
  return stitcher->stitch(fwdGraphId, bwdGraphInfo, stitchIndices);
}

namespace {
bool init = Transform::registerTransform(new Autodiff);
}

} // namespace popart
