// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <transforms/autodiff/stitcherfactory.hpp>

#include <transforms/autodiff/addfwdoutputstitcher.hpp>
#include <transforms/autodiff/autodiffiradapter.hpp>
#include <transforms/autodiff/recomputestitcher.hpp>
#include <transforms/autodiff/safeaddfwdoutputstitcher.hpp>

namespace popart {

std::unique_ptr<StitcherInterface>
StitcherFactory::createStitcher(AutodiffIrInterface &ir,
                                AutodiffStitchStrategy stitchStrategy) {

  switch (stitchStrategy) {
  case AutodiffStitchStrategy::RecomputeMinimal: {
    return std::make_unique<RecomputeStitcher>(
        ir, RecomputeStitcher::StitchIndexMode::Minimal);
    break;
  }
  case AutodiffStitchStrategy::RecomputeAllNonInputs: {
    return std::make_unique<RecomputeStitcher>(
        ir, RecomputeStitcher::StitchIndexMode::AllNonInputs);
    break;
  }
  case AutodiffStitchStrategy::AddFwdOutputs: {
    return std::make_unique<AddFwdOutputStitcher>(ir);
    break;
  }
  case AutodiffStitchStrategy::SafeAddFwdOutputs: {
    return std::make_unique<SafeAddFwdOutputStitcher>(ir);
    break;
  }
  default: {
    throw error("[Autodiff] Unknown value for AutodiffStitchStrategy ({})",
                static_cast<int>(stitchStrategy));
  }
  }
}

} // namespace popart
