#include <poponnx/error.hpp>
#include <poponnx/half.hpp>

namespace poponnx {

Half Half::operator+(const Half &) {
  throw error("Half::operator+ is not implemented");
}

} // namespace poponnx
