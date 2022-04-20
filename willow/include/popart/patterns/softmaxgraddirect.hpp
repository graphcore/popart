// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_SOFTMAXGRADDIRECT_HPP
#define GUARD_NEURALNET_SOFTMAXGRADDIRECT_HPP

#include <popart/patterns/fuser.hpp>

#include "popart/names.hpp"

namespace popart {
class Op;
struct OperatorIdentifier;

// consider,
// (label), (probs), (d_in) -> [NLLGrad]
// [NllGrad] -> (d_probs)
// (d_probs), (probs) -> [SoftmaxGrad] -> (d_acts).
// This pattern replaces this with,
// (label), (probs), (d_in) -> [SoftmaxGradDirect] -> (d_acts).

class SoftmaxGradDirect : public Fuser {
private:
  const OperatorIdentifier &get0() const final;
  const OperatorIdentifier &get1() const final;
  OpId moveMergedIntoIr(Op *baseOp) const final;
};

} // namespace popart

#endif
