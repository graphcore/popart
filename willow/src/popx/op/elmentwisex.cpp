#include <poponnx/op/elementwise.hpp>
#include <poponnx/popx/op/elementwisex.hpp>

namespace poponnx {
namespace popx {

ElementWiseUnaryOpx::ElementWiseUnaryOpx(Op *op, Devicex *devicex)
    : Opx(op, devicex) {}

InputCreatorType ElementWiseUnaryOpx::getInputCreatorType(InIndex) const {
  return InputCreatorType::AGNOSTICTOLAYOUT;
}

} // namespace popx
} // namespace poponnx
