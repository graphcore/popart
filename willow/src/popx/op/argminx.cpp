#include <poponnx/op/argmin.hpp>
#include <poponnx/popx/devicex.hpp>
#include <poponnx/popx/op/argminx.hpp>
#include <poponnx/popx/opxmanager.hpp>

namespace poponnx {
namespace popx {

poplar::Tensor ArgMinOpx::selectSlice(const poplar::Tensor &sorted,
                                      unsigned axis) const {
  // Take the first (minimum) slice
  return sorted.slice(0, 1, axis);
}

namespace {
OpxCreator<ArgMinOpx> argMinOpxCreator(Onnx::Operators::ArgMin_1);
} // namespace

} // namespace popx
} // namespace poponnx
