// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_SGD0COMBO_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_SGD0COMBO_HPP_

#include <map>
#include <memory>
#include <set>
#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/op/varupdate.hpp>

#include "popart/datatype.hpp"
#include "popart/optimizervalue.hpp"

namespace popart {
class OpSerialiserBase;

enum class OptimizerReductionType;

/**
 * \brief A single Op that encapsulates all the information needed to describe
 * an SGD0 optimiser step.
 *
 * The "0" in the name signifies that there is no optimizer state (note a
 * gradient accum tensor may still be required) \sa SGD for the definition of
 * what SGD0 is.
 *
 * The "Combo" in the name signifies that this Op will later be decomposed into
 * many Ops and Tensors that actually implement the optimiser step. In this
 * case, by the SGD0Decompose pattern. \sa SGD0Decompose for the definition of
 * this decomposition.
 */
class SGD0ComboOp final : public VarUpdateWithUpdaterOp {
public:
  SGD0ComboOp(OptimizerValue initialSwd,
              OptimizerValue initialSlr,
              bool withGradAccum_,
              OptimizerReductionType reductionType_,
              DataType accumType_,
              const Op::Settings &);

  OptimizerValue initSlr0;
  OptimizerValue initWdsf0;

  // Gradient accumulation
  const bool withGradAccum;

  const OptimizerReductionType reductionType;

  // Data type of accumulator and momentum
  const DataType accumType;

  // If the scaled learning rate is not constant, this is the index at which it
  // will be consumed by this Op
  static InIndex getSlr0InIndex() { return 2; }

  // If the weight decay scale factor is not constant, this is the index at
  // which it will be consumed by this Op
  static InIndex getWdsf0InIndex() { return 3; }

  std::unique_ptr<Op> clone() const final;

  std::set<InIndex> optionalInputs() const override;

  // map of size 0/1/2, containing all non-const optimizer Tensors for this Op
  std::map<InIndex, TensorId> optimizerInputs() const override;

  void appendOutlineAttributes(OpSerialiserBase &) const override;

  // this Op should not be present when outlining is performed
  float getSubgraphValue() const override { return -1.0f; }
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_SGD0COMBO_HPP_
