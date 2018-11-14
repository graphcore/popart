#include <algorithm>
#include <poponnx/error.hpp>
#include <poponnx/pad.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/util.hpp>

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

PadOp::PadOp(const onnx::NodeProto &node, Ir *_pir) : Op(node, _pir) {
  nAtts.set(pads, "pads");
}

bool PadOp::padSizeZero() const {
  return std::all_of(
      pads.cbegin(), pads.cend(), [](int64_t p) { return p == 0; });
}

} // namespace willow
