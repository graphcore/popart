#ifndef GUARD_NEURALNET_SGD1DECOMPOSE_PATTERN_HPP
#define GUARD_NEURALNET_SGD1DECOMPOSE_PATTERN_HPP

#include <popart/patterns/patterns.hpp>

namespace popart {

class SGD1Decompose : public PreAliasPattern {
public:
  bool matches(Op *) const final;
  std::vector<const Tensor *> touches(Op *) const final;
  bool apply(Op *) const final;
};

} // namespace popart

#endif
