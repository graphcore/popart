#ifndef GUARD_NEURALNET_INPLACE_HPP
#define GUARD_NEURALNET_INPLACE_HPP

#include <poponnx/patterns/pattern.hpp>

namespace poponnx {

// for all cases with 0 outputs
class Inplace : public Pattern {
public:
  bool matches(Op *op) const final;
  // which input indices does this Inplace Pattern target?
  virtual std::vector<InIndex> targetInIndices(Op *) const = 0;
  std::vector<const Tensor *> touches(Op *op) const final;
  bool apply(Op *op) const final;
  PatternPhase phase() const final { return PatternPhase::WITHTOPOCONS; }
};

// For an Op "op" with
//    N inputs and 1 output,
// replace it with an in-place Op with
//    N inputs and 0 outputs,
// with the in-place Op modifying input 0 to have the
// value that the output of op would have had.
class Inplace0 : public Inplace {
public:
  virtual std::vector<InIndex> targetInIndices(Op *) const final { return {0}; }
};

class InplaceAll : public Inplace {
public:
  virtual std::vector<InIndex> targetInIndices(Op *op) const final {
    std::vector<InIndex> indices;
    for (auto &x : op->input->tensorMap()) {
      indices.push_back(x.first);
    }
    return indices;
  }
};

} // namespace poponnx

#endif
