#include <popops/Pad.hpp>
#include <poponnx/error.hpp>
#include <poponnx/op/pad.hpp>
#include <poponnx/popx/op/padgradx.hpp>
#include <poponnx/popx/op/padx.hpp>
#include <poponnx/popx/opxmanager.hpp>

namespace poponnx {
namespace popx {

PadOpx::PadOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<PadOp>(op);
}

void PadOpx::grow(poplar::program::Sequence &) const {
  auto pad_op    = getPadOp();
  auto &&pads    = pad_op->getPads();
  auto pad_value = pad_op->getPadValue();
  auto t_rank    = inInfo(PadOp::getInIndex()).rank();
  auto &&mode    = pad_op->getMode();

  std::vector<std::ptrdiff_t> lower_padding;
  std::vector<std::ptrdiff_t> upper_padding;
  for (int i = 0; i < t_rank; i++) {
    lower_padding.push_back(pads[i]);
    upper_padding.push_back(pads[t_rank + i]);
  }

  poplar::Tensor out_tensor;

  if (mode == "constant") {
    out_tensor = popops::pad(graph(),
                             get(inId(PadOp::getInIndex())),
                             lower_padding,
                             upper_padding,
                             pad_value);
  } else if (mode == "edge") {
    out_tensor = popops::pad(get(inId(PadOp::getInIndex())),
                             lower_padding,
                             upper_padding,
                             popops::padding::Type::EDGE);
  } else if (mode == "reflect") {
    out_tensor = popops::pad(get(inId(PadOp::getInIndex())),
                             lower_padding,
                             upper_padding,
                             popops::padding::Type::REFLECT);
  } else {
    throw error("Bad mode type `{}' passed to pad op", mode);
  }

  insert(outId(PadOp::getOutIndex()), out_tensor);
}

PadOp *PadOpx::getPadOp() const { return dynamic_cast<PadOp *>(op_p); }

PadGradOpx::PadGradOpx(Op *op, Devicex *devicex) : SliceOpx(op, devicex) {
  verifyOp<PadGradOpx>(op, Onnx::GradOperators::PadGrad);
}

namespace {
OpxCreator<PadOpx> padOpxCreator(Onnx::Operators::Pad_2);
OpxCreator<PadGradOpx> padGradOpxCreator(Onnx::GradOperators::PadGrad);
} // namespace

} // namespace popx
} // namespace poponnx
