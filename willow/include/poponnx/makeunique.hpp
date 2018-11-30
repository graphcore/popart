#ifndef GUARD_NEURALNET_MAKEUNIQUE_HPP
#define GUARD_NEURALNET_MAKEUNIQUE_HPP

#include <memory>

namespace poponnx {

// TODO : If we move to C++14, this function will be standard.
template <typename T, typename... Args>
std::unique_ptr<T> make_unique(Args &&... args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

} // namespace poponnx

#endif
