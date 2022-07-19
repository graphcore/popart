// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_SRC_STOCHASTICROUNDINGASSUMPTIONVERIFIER_HPP_
#define POPART_WILLOW_SRC_STOCHASTICROUNDINGASSUMPTIONVERIFIER_HPP_

#include <functional>

namespace popart {

// Forward declare.
class Ir;

/**
 * Class to check IR assumptions pertaining to stochastic rounding.
 *
 * NOTE: I am deliberately not defining this as a method of popart::Ir to avoid
 * making that class bigger.
 **/
class StochasticRoundingAssumptionVerifier {
public:
  /**
   * Construct an `StochasticRoundingAssumptionVerifier`.
   * @param ir a reference to the IR object to check.
   */
  StochasticRoundingAssumptionVerifier(const Ir &ir);

  /**
   * Check whether 1) if stochastic rounding is disabled no ops have the
   * `stochasticRoundingMethod` attribute set and 2) if stochastic rounding is
   * enabled that all ops have the `stochasticRoundingMethod` attribute set.
   *
   * If either condition does not hold a popart::error is thrown.
   */
  virtual void verify();

public:
  // The IR.
  std::reference_wrapper<const Ir> ir;
};

} // namespace popart

#endif // POPART_WILLOW_SRC_STOCHASTICROUNDINGASSUMPTIONVERIFIER_HPP_
