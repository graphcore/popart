// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <transforms/autodiff/stitcherfactory.hpp>

#include <transforms/autodiff/addfwdoutputstitcher.hpp>
#include <transforms/autodiff/autodiffiradapter.hpp>
#include <transforms/autodiff/recomputestitcher.hpp>

namespace popart {

std::unique_ptr<StitcherInterface>
StitcherFactory::createStitcher(AutodiffIrInterface &ir,
                                Autodiff::StitchStrategy stitchStrategy) {

  switch (stitchStrategy) {
  case Autodiff::StitchStrategy::RecomputeMinimal: {
    return std::make_unique<RecomputeStitcher>(
        ir, RecomputeStitcher::StitchIndexMode::Minimum);
    break;
  }
  case Autodiff::StitchStrategy::RecomputeAllNonInputs: {
    return std::make_unique<RecomputeStitcher>(
        ir, RecomputeStitcher::StitchIndexMode::AllNonInputs);
    break;
  }
  case Autodiff::StitchStrategy::AddFwdOutputs: {
    return std::make_unique<AddFwdOutputStitcher>(ir);
    break;
  }
  default: {
    throw error("[Autodiff] Unknown value for StitchStrategy ({})",
                static_cast<int>(stitchStrategy));
  }
  }
}

} // namespace popart
