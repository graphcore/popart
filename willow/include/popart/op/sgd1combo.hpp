// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_SGD1VARUPDATECOMBOOP_HPP
#define GUARD_NEURALNET_SGD1VARUPDATECOMBOOP_HPP

#include <popart/op/sgdcombobase.hpp>
#include <popart/optimizer.hpp>
#include <popart/optimizervalue.hpp>

namespace popart {

// The "1" in the name signifies that there is 1 persistant Tensor required and
// assocatiated to the Variable Tensor being updated. This is the Op generated
// if gradient accumulation is used, or if there is non-zero momentum term for
// the Variable Tensor being updated.

// The "Combo" in the name signfies that this Op will be decomposed into 3
// smaller Ops : (1) SGD1AccumlateOp (2) SGD1VarUpdateOp (3) SGD1AcclUpdateOp

class SGD1ComboOp final : public SGDComboBaseOp {
public:
  using SGDComboBaseOp::SGDComboBaseOp;

  std::unique_ptr<Op> clone() const final;
};

} // namespace popart

#endif
