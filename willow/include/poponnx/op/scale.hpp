#ifndef GUARD_NEURALNET_SCALE_HPP
#define GUARD_NEURALNET_SCALE_HPP

#include <poponnx/op.hpp>

namespace poponnx {

// y = scale_factor * x
class ScaleOp : public Op {
public:
  ScaleOp(const OpConstructorBundle &, float scale_factor);
  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  void setup() final;

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }

  float getScaleFactor() const;

private:
  float scale_factor;
};

} // namespace poponnx

#endif
