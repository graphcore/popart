#ifndef GUARD_NEURALNET_SOFTMAXGRADDIRECT_HPP
#define GUARD_NEURALNET_SOFTMAXGRADDIRECT_HPP

#include <poponnx/patterns/fuser.hpp>

namespace willow {

// consider,
// (label), (probs) -> [NLLGrad]
// [NllGrad] -> (d_probs)
// (d_probs), (probs) -> [SoftmaxGrad] -> (d_acts).
// This pattern replaces this with,
// (label), (probs) -> [SoftmaxGradDirect] -> (d_acts).

class SoftmaxGradDirect : public Fuser {
private:
  OpType get0() const final;
  OpType get1() const final;
  OpId moveMergedIntoIr(Op *baseOp) const final;
};

} // namespace willow

#endif
