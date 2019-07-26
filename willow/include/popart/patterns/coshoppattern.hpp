#ifndef GUARD_NEURALNET_COSH_OP_PATTERN_HPP
#define GUARD_NEURALNET_COSH_OP_PATTERN_HPP

#include <popart/patterns/patterns.hpp>

namespace popart {

// Replace CoshOp
// cosh(x) = (exp(x) + exp(-x))/2
class CoshOpPattern : public PreAliasPattern {
public:
  bool matches(Op *) const override;
  std::vector<const Tensor *> touches(Op *) const override;
  bool apply(Op *) const override;
};

} // namespace popart

#endif
