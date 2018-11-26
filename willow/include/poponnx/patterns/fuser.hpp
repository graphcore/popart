#ifndef GUARD_NEURALNET_FUSER_HPP
#define GUARD_NEURALNET_FUSER_HPP

#include <poponnx/patterns/patterns.hpp>

namespace willow {

// {(a), (b), (c)} -> [op0] -> (d) -> [op1] -> {(e), (f)}
//                    ====================>
// {(a), (b), (c)} ->        [op01]         -> {(e), (f)}
class Fuser : public Pattern {
public:
  bool matches(Op *) const final;
  // Only (d) is touched. Therefore, a Pattern where [op1] and
  // [op01] perform inplace changes to an input tensor should
  // not inherit from Fuser.
  std::vector<const Tensor *> touches(Op *) const final;
  void apply(Op *) const final;
  PatternPhase phase() const final { return PatternPhase::PRETOPOCONS; }

private:
  // OpType of op0 in schematic
  virtual OpType get0() const = 0;
  // OpType of op1 in schematic
  virtual OpType get1() const = 0;
  // how to create a new op01 and move it into Ir
  virtual OpId moveMergedIntoIr(Op *baseOp) const = 0;
};

} // namespace willow

#endif
