#include <algorithm>
#include <willow/error.hpp>
#include <willow/pad.hpp>
#include <willow/tensor.hpp>
#include <willow/util.hpp>

namespace willow {

std::unique_ptr<Op> PadOp::clone() const {
  return std::unique_ptr<Op>(new PadOp(*this));
}

void PadOp::setup() {

  int tRank = input.tensor(0)->info.rank();
  if (pads.size() != 2 * tRank) {
    throw error("Tensor rank not twice padding size");
  }

  std::vector<int64_t> outShape(tRank, 0);
  for (int i = 0; i < input.tensor(0)->info.rank(); ++i) {
    outShape[i] = input.tensor(0)->info.dim(i) + pads[i] + pads[i + tRank];
  }

  output.tensor(0)->info = {input.tensor(0)->info.dataType(), outShape};
}

PadOp::PadOp(const onnx::NodeProto &node, Ir *pir) : Op(node, pir) {
  nAtts.set(pads, "pads");
}

bool PadOp::padSizeZero() const {
  return std::all_of(
      pads.cbegin(), pads.cend(), [](int64_t p) { return p == 0; });
}

} // namespace willow
