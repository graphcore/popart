#include <poponnx/error.hpp>
#include <poponnx/ndindices.hpp>
#include <poponnx/tensorinfo.hpp>

namespace poponnx {

NDIndices::NDIndices(const TensorInfo &i) : info(i) {}

std::vector<int64_t> NDIndices::unflatten(int64_t rem) {
  std::vector<int64_t> indices;
  for (int64_t d : info.shape()) {
    indices.push_back(rem % d);
    rem /= d;
  }
  if (rem != 0) {
    throw error("index too large in unflatten");
  }
  return indices;
}

int64_t NDIndices::flatten(std::vector<int64_t> indices) {
  // case 2) bailing
  if (indices.size() < info.rank()) {
    throw error("too few indices in flatten argument");
  }
  // case 1) ignoring left-most (slowest) indices of indices
  else if (indices.size() > info.rank()) {
    indices = std::vector<int64_t>(indices.end() - info.rank(), indices.end());
  }
  int64_t stride = info.nelms();
  int64_t index  = 0l;
  for (int d = 0; d < info.rank(); ++d) {
    stride /= info.dim(d);
    // 3) modulo arithmetic
    index += (indices[d] % info.dim(d)) * stride;
  }
  if (index > info.nelms()) {
    throw error("ILE in flatten : final index too large");
  }
  return index;
}

} // namespace poponnx
