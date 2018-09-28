#include <neuralnet/pad.hpp>
#include <neuralnet/tensor.hpp>

namespace neuralnet {
PadOp::PadOp(const onnx::NodeProto &node, Graph *pgraph) : Op(node, pgraph) {
  nAtts.set(pads, "pads");
}

bool PadOp::padSizeZero() const {
  return std::all_of(
      pads.cbegin(), pads.cend(), [](int64_t p) { return p == 0; });
}

} // namespace neuralnet
