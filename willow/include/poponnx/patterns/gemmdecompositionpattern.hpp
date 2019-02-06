#ifndef GUARD_NEURALNET_GEMM_DECOMPOSITION_PATTERN_HPP
#define GUARD_NEURALNET_GEMM_DECOMPOSITION_PATTERN_HPP

#include <poponnx/patterns/pattern.hpp>
#include <poponnx/patterns/patterns.hpp>

namespace poponnx {

// Replace GemmOp with
// gemm(a, b, c) = add(scale(matmul(a, b), alpha), scale(c, beta))
class GemmDecompositionPattern : public PreAliasPattern {
public:
  bool matches(Op *) const override;
  std::vector<const Tensor *> touches(Op *) const override;
  bool apply(Op *) const override;

  void
  transposeTensor(const TensorId &input, const TensorId &output, Op *op) const;
  void scaleTensor(const TensorId &input,
                   const TensorId &output,
                   float scale_factor,
                   Op *op) const;
};

} // namespace poponnx

#endif
