#ifndef GUARD_NEURALNET_ARGEXTREMA_HPP
#define GUARD_NEURALNET_ARGEXTREMA_HPP

#include <poponnx/op.hpp>

namespace poponnx {

// The base class for an op that wants to choose some extreme values from an
// input tensor.
class ArgExtremaOp : public Op {
public:
  ArgExtremaOp(const OperatorIdentifier &_opid,
               int64_t axis,
               int64_t keepdims,
               const Op::Settings &settings);
  std::unique_ptr<Op> clone() const override;
  void setup() final;

  int64_t getKeepDims() const;
  int64_t getAxis() const;

  void appendAttributes(OpSerialiserBase &) const final;

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }

private:
  void validateAxis() const;

  const int64_t keepdims;
  const int64_t axis;
};

} // namespace poponnx

#endif
