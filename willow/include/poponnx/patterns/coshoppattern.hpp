#ifndef GUARD_NEURALNET_COSH_OP_PATTERN_HPP
#define GUARD_NEURALNET_COSH_OP_PATTERN_HPP

#include <poponnx/patterns/patterns.hpp>

namespace poponnx {

// Replace CoshOp
// cosh(x) = (exp(x) + exp(-x))/2
class CoshOpPattern : public Pattern {
public:
  bool matches(Op *) const override;
  std::vector<const Tensor *> touches(Op *) const override;
  bool apply(Op *) const override;
  PatternPhase phase() const final { return PatternPhase::PRETOPOCONS; }
};

} // namespace poponnx

#endif
