#include <poponnx/error.hpp>
#include <poponnx/logging.hpp>

#include <poplar/exceptions.hpp>
#include <poputil/exceptions.hpp>

namespace willow {

error::error(const std::string &what) : std::runtime_error(what) {
  logging::err(what);
}

ErrorSource getErrorSource(const std::exception &e) {
  if (dynamic_cast<const willow::error *>(&e)) {
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

} // namespace willow
