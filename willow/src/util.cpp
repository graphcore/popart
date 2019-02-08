#include <poponnx/names.hpp>
#include <poponnx/util.hpp>

namespace std {
std::ostream &operator<<(std::ostream &ss, const std::vector<int64_t> &v) {
  poponnx::appendSequence(ss, v);
  return ss;
}
} // namespace std

namespace poponnx {

std::ostream &operator<<(std::ostream &ss, const std::vector<std::size_t> &v) {
  appendSequence(ss, v);
  return ss;
}
} // namespace poponnx
