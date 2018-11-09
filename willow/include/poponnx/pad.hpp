#ifndef GUARD_NEURALNET_PAD_HPP
#define GUARD_NEURALNET_PAD_HPP

#include <poponnx/ir.hpp>

namespace willow {

class PadOp : public Op {
public:
  PadOp(const onnx::NodeProto &node, Ir *pir);
  virtual std::unique_ptr<Op> clone() const override final;
  // returns true of all pad size in all dimensions
  // and on both sides, are zero
  bool padSizeZero() const;
  void setup() override final;

private:
  std::vector<int64_t> pads;
};
} // namespace willow

#endif
