// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <popart/transforms/autodiff.hpp>

#include <popart/graph.hpp>

#include <transforms/autodiff/autodiffiradapter.hpp>
#include <transforms/autodiff/gradgrowerloss.hpp>
#include <transforms/autodiff/gradgrowermaingraph.hpp>
#include <transforms/autodiff/gradgrowerop.hpp>
#include <transforms/autodiff/gradgrowersumop.hpp>

namespace popart {

std::size_t Autodiff::id() { return typeid(Autodiff).hash_code(); }

bool Autodiff::apply(Graph &graph) const {

  AutodiffIrAdapter irAdapter(graph.getIr());
  auto gradGrowerOp    = std::make_unique<GradGrowerOp>(irAdapter);
  auto gradGrowerLoss  = std::make_unique<GradGrowerLoss>(irAdapter);
  auto gradGrowerSumOp = std::make_unique<GradGrowerSumOp>(irAdapter);
  GradGrowerMainGraph gradMainGraphGrower(irAdapter,
                                          std::move(gradGrowerOp),
                                          std::move(gradGrowerLoss),
                                          std::move(gradGrowerSumOp));
  gradMainGraphGrower.growGradMainGraph();

  return true;
}

namespace {
bool init = Transform::registerTransform(new Autodiff);
}

} // namespace popart
