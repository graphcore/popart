#include <algorithm>
#include <willow/pad.hpp>
#include <willow/tensor.hpp>

namespace willow {

std::unique_ptr<Op> PadOp::clone() const {
  return std::unique_ptr<Op>(new PadOp(*this));
}

PadOp::PadOp(const onnx::NodeProto &node, Ir *pir) : Op(node, pir) {
  nAtts.set(pads, "pads");
}

bool PadOp::padSizeZero() const {
  return std::all_of(
      pads.cbegin(), pads.cend(), [](int64_t p) { return p == 0; });
}

} // namespace willow
