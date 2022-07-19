// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_PATTERNS_SOFTMAXGRADDIRECT_HPP_
#define POPART_WILLOW_INCLUDE_POPART_PATTERNS_SOFTMAXGRADDIRECT_HPP_

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

#endif // POPART_WILLOW_INCLUDE_POPART_PATTERNS_SOFTMAXGRADDIRECT_HPP_
