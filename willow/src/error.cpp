#include <poponnx/error.hpp>
#include <poponnx/logging.hpp>

#include <poplar/exceptions.hpp>
#include <poputil/exceptions.hpp>

namespace poponnx {

ErrorSource getErrorSource(const std::exception &e) {
  if (dynamic_cast<const poponnx::memory_allocation_err *>(&e)) {
    return ErrorSource::poponnx;
  }
  if (dynamic_cast<const poponnx::error *>(&e)) {
    return ErrorSource::poponnx;
  }
  if (dynamic_cast<const poplar::poplar_error *>(&e)) {
    return ErrorSource::poplar;
  }
  if (dynamic_cast<const poputil::poplibs_error *>(&e)) {
    return ErrorSource::poplibs;
  }
  return ErrorSource::unknown;
}

} // namespace poponnx
