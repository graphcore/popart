// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <stochasticroundingassumptionverifier.hpp>
#include <vector>
#include <popart/ir.hpp>

#include "popart/error.hpp"
#include "popart/logging.hpp"
#include "popart/op.hpp"
#include "popart/sessionoptions.hpp"

namespace popart {

StochasticRoundingAssumptionVerifier::StochasticRoundingAssumptionVerifier(
    const Ir &ir_)
    : ir{ir_} {}

void StochasticRoundingAssumptionVerifier::verify() {
  // NOTE: We assume at time of lowering Ops have the stochasticRoundingMethod
  // set if and only if the 'enableStochasticRounding' method is set in the
  // session options. We check this assumption here and throw and error when it
  // does not hold.
  bool srSessionOption = ir.get().getSessionOptions().enableStochasticRounding;
  for (Op *op : ir.get().getAllOps()) {
    bool opHasSr = op->hasStochasticRoundingMethod();

    if (srSessionOption && !opHasSr) {
      throw error("[StochasticRoundingAssumptionVerifier] PopART assumes that "
                  "if the session option 'enableStochasticRounding' is set "
                  "then at time of lowering every Op has the "
                  "'stochasticRoundingMethod' set, but this is not the case "
                  "for '{}'",
                  op->debugName());
    } else if (!srSessionOption && opHasSr) {
      throw error("[StochasticRoundingAssumptionVerifier] PopART assumes that "
                  "if the session option 'enableStochasticRounding' is not set "
                  "then at time of lowering no Op has the "
                  "'stochasticRoundingMethod' set, but this is not the case "
                  "for '{}'",
                  op->debugName());
    }
  }
}

} // namespace popart
