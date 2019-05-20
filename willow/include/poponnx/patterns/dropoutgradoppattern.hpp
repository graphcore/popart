#ifndef GUARD_NEURALNET_DROPOUT_GRAD_OP_PATTERN_HPP
#define GUARD_NEURALNET_DROPOUT_GRAD_OP_PATTERN_HPP

#include <poponnx/patterns/patterns.hpp>

namespace poponnx {

// Replace a DropoutGradOp with
// (grad_in) -> [Dropout] -> (grad_out)
class DropoutGradOpPattern : public PreAliasPattern {
public:
  bool matches(Op *) const override;
  std::vector<const Tensor *> touches(Op *) const override;
  bool apply(Op *) const override;
};

} // namespace poponnx

#endif
