#ifndef GUARD_NEURALNET_TANH_GRAD_OP_PATTERN_HPP
#define GUARD_NEURALNET_TANH_GRAD_OP_PATTERN_HPP

#include <poponnx/patterns/patterns.hpp>

namespace poponnx {

// Replace TanhGradOp
// grad_out = grad_in / square(cosh(fwd_in))
class TanhGradOpPattern : public Pattern {
public:
  bool matches(Op *) const override;
  std::vector<const Tensor *> touches(Op *) const override;
  bool apply(Op *) const override;
  PatternPhase phase() const final { return PatternPhase::PRETOPOCONS; }
};

} // namespace poponnx

#endif
