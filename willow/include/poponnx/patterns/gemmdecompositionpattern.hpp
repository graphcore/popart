#ifndef GUARD_NEURALNET_GEMM_DECOMPOSITION_PATTERN_HPP
#define GUARD_NEURALNET_GEMM_DECOMPOSITION_PATTERN_HPP

#include <poponnx/patterns/patterns.hpp>

namespace poponnx {

// Replace GemmOp with
// gemm(a, b, c) = add(scale(matmul(a, b), alpha), scale(c, beta))
class GemmDecompositionPattern : public Pattern {
public:
  bool matches(Op *) const override;
  std::vector<const Tensor *> touches(Op *) const override;
  bool apply(Op *) const override;
  PatternPhase phase() const final { return PatternPhase::PRETOPOCONS; }
};

} // namespace poponnx

#endif
