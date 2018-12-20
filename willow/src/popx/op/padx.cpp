#include <poponnx/error.hpp>
#include <poponnx/op/pad.hpp>
#include <poponnx/popx/op/padx.hpp>
#include <poponnx/popx/opxmanager.hpp>

namespace poponnx {
namespace popx {

PadOpx::PadOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<PadOp>(op, Onnx::Operators::Pad);
}

namespace {
OpxCreator<PadOpx> padOpxCreator(Onnx::Operators::Pad);
}

} // namespace popx
} // namespace poponnx
