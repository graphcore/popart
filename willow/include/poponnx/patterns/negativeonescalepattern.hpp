#ifndef GUARD_NEURALNET_NEGATIVE_ONE_SCALE_PATTERN_HPP
#define GUARD_NEURALNET_NEGATIVE_ONE_SCALE_PATTERN_HPP

#include <poponnx/patterns/patterns.hpp>
#include <poponnx/patterns/sequenceexpander.hpp>

namespace poponnx {

class NegativeOneScalePattern : public SequenceExpander {
public:
  bool matches(Op *) const override;

private:
  std::vector<std::unique_ptr<Op>> sequence(Op *op) const final;
};

} // namespace poponnx

#endif
