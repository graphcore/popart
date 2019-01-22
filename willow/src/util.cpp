#include <poponnx/util.hpp>

namespace poponnx {

std::ostream &operator<<(std::ostream &ss, const std::vector<int64_t> &v) {
  appendSequence(ss, v);
  return ss;
}

} // namespace poponnx
