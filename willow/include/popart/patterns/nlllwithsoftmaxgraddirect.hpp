// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_PATTERNS_NLLLWITHSOFTMAXGRADDIRECT_HPP_
#define POPART_WILLOW_INCLUDE_POPART_PATTERNS_NLLLWITHSOFTMAXGRADDIRECT_HPP_

#include <vector>
#include <popart/patterns/pattern.hpp>

namespace popart {
class Op;
class Tensor;

// consider,
// (label), (probs) -> [NllOp]
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

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_PATTERNS_NLLLWITHSOFTMAXGRADDIRECT_HPP_
