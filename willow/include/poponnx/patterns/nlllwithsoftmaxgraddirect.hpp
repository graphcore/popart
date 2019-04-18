#ifndef GUARD_NEURALNET_NLLLWITHSOFTMAXGRADDIRECT_HPP
#define GUARD_NEURALNET_NLLLWITHSOFTMAXGRADDIRECT_HPP

#include <poponnx/patterns/pattern.hpp>

namespace poponnx {

// consider,
// (label), (probs) -> [NllLoss]
// [NllGrad] -> (loss)
// (label), (probs) -> [SoftmaxGradDirect] -> (d_acts).
// This pattern replaces this with,
// (label), (probs) -> [NlllWithSoftmaxGradDirect] -> (loss), (d_acts).

class NlllWithSoftmaxGradDirect : public PreAliasPattern {
public:
  bool matches(Op *) const override;
  std::vector<const Tensor *> touches(Op *) const override;
  bool apply(Op *) const override;
};

} // namespace poponnx

#endif
