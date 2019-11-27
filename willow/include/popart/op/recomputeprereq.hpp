#ifndef GUARD_NEURALNET_RECOMPUTEPREREQ_HPP
#define GUARD_NEURALNET_RECOMPUTEPREREQ_HPP

#include <popart/op.hpp>

namespace popart {

// Op that ensures the tensors needed by recomputing operations are loaded by
// a CacheLoad operation before they are used.
class RecomputePrereqOp : public Op {
public:
  RecomputePrereqOp(Settings settings)
      : Op(Onnx::CustomOperators::RecomputePrereq, settings) {}
  std::unique_ptr<Op> clone() const override {
    return std::make_unique<RecomputePrereqOp>(*this);
  }
  void setup() final {}
  float getSubgraphValue() const final { return 0.0f; }
  bool isOutlineable() const override { return false; }
};

} // namespace popart

#endif
