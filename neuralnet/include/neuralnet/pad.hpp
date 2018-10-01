#ifndef GUARD_NEURALNET_PAD_HPP
#define GUARD_NEURALNET_PAD_HPP

#include <neuralnet/graph.hpp>

namespace neuralnet {

class PadOp : public Op {
public:
  PadOp(const onnx::NodeProto &node, Graph *pgraph);
  virtual std::unique_ptr<Op> clone() const override final;
  // returns true of all pad size in all dimensions
  // and on both sides, are zero
  bool padSizeZero() const;

private:
  std::vector<int64_t> pads;
};
} // namespace neuralnet

#endif
