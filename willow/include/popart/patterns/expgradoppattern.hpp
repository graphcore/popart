#ifndef GUARD_NEURALNET_EXP_GRAD_OP_PATTERN_HPP
#define GUARD_NEURALNET_EXP_GRAD_OP_PATTERN_HPP

#include <popart/patterns/patterns.hpp>

namespace popart {

// Replace a ExpGradOp with
// grad_out = grad_in * fwd_out
class ExpGradOpPattern : public PreAliasPattern {
public:
  bool matches(Op *) const override;
  std::vector<const Tensor *> touches(Op *) const override;
  bool apply(Op *) const override;
};

} // namespace popart

#endif
