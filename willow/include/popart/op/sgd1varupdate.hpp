// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_SGD1VARUPDATE_HPP
#define GUARD_NEURALNET_SGD1VARUPDATE_HPP

#include <popart/op/varupdate.hpp>
#include <popart/optimizervalue.hpp>

namespace popart {

/**
 * @brief Performs the SGD1 weight update equation.
 *
 * Let:
 *   `w` be the input at `getVarToUpdateInIndex()`
 *   `g` be the input at `getUpdaterInIndex()`
 * then this op performs:
 *   w <- w - slr1 * g
 *
 * \sa SGD for how this is derived and the definition of slr1.
 */
class SGD1VarUpdateOp : public VarUpdateWithUpdaterOp {
public:
  SGD1VarUpdateOp(OptimizerValue initSlr1, const Op::Settings &);

  std::unique_ptr<Op> clone() const override;
  std::map<InIndex, TensorId> optimizerInputs() const final;
  void appendOutlineAttributes(OpSerialiserBase &) const final;

  const OptimizerValue initSlr1;
  static InIndex getSlr1InIndex() { return 2; }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }
};

} // namespace popart

#endif
