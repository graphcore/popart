#ifndef GUARD_NEURALNET_INPLACE_HPP
#define GUARD_NEURALNET_INPLACE_HPP

#include <poponnx/patterns/patterns.hpp>

namespace poponnx {

// For an Op "op" with
//    N inputs and 1 output,
// replace it with an in-place Op with
//    N inputs and 0 outputs,
// with the in-place Op modifying input 0 to have the
// value that the output of op would have had.
class Inplace0 : public Pattern {
public:
  bool matches(Op *op) const final;
  std::vector<const Tensor *> touches(Op *op) const final;
  bool apply(Op *op) const final;
  PatternPhase phase() const final { return PatternPhase::WITHTOPOCONS; }
};

} // namespace poponnx

#endif
