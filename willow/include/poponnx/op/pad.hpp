#ifndef GUARD_NEURALNET_PAD_HPP
#define GUARD_NEURALNET_PAD_HPP

#include <poponnx/op.hpp>

namespace poponnx {

class PadOp : public Op {
public:
  PadOp(const onnx::NodeProto &node, Ir *pir);
  std::unique_ptr<Op> clone() const final;
  // returns true of all pad size in all dimensions
  // and on both sides, are zero
  bool padSizeZero() const;
  void setup() final;

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }

private:
  std::vector<int64_t> pads;
};
} // namespace poponnx

#endif
