#ifndef GUARD_NEURALNET_SOFTMAXGRADDIRECT_HPP
#define GUARD_NEURALNET_SOFTMAXGRADDIRECT_HPP

#include <popart/patterns/fuser.hpp>

namespace popart {

// consider,
// (label), (probs) -> [NLLGrad]
// [NllGrad] -> (d_probs)
// (d_probs), (probs) -> [SoftmaxGrad] -> (d_acts).
// This pattern replaces this with,
// (label), (probs) -> [SoftmaxGradDirect] -> (d_acts).

class SoftmaxGradDirect : public Fuser {
private:
  const OperatorIdentifier &get0() const final;
  const OperatorIdentifier &get1() const final;
  OpId moveMergedIntoIr(Op *baseOp) const final;
};

} // namespace popart

#endif
