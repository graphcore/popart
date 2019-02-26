#include <poponnx/error.hpp>
#include <poponnx/op/argmax.hpp>
#include <poponnx/popx/devicex.hpp>
#include <poponnx/popx/op/argmaxx.hpp>
#include <poponnx/popx/opxmanager.hpp>

namespace poponnx {
namespace popx {

poplar::Tensor ArgMaxOpx::selectSlice(const poplar::Tensor &sorted,
                                      unsigned axis) const {
  const auto size = sorted.dim(axis);

  // Take the last (maximum) slice
  return sorted.slice(size - 1, size, axis);
}

namespace {
OpxCreator<ArgMaxOpx> argMaxOpxCreator(Onnx::Operators::ArgMax_1);
} // namespace

} // namespace popx
} // namespace poponnx
