// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <popart/error.hpp>
#include <popart/logging.hpp>

#include <poplar/exceptions.hpp>
#include <poputil/exceptions.hpp>

namespace popart {

ErrorSource getErrorSource(const std::exception &e) {
  if (dynamic_cast<const popart::internal_error *>(&e)) {
    return ErrorSource::popart_internal;
  }
  if (dynamic_cast<const popart::memory_allocation_err *>(&e)) {
    return ErrorSource::popart;
  }
  if (dynamic_cast<const popart::error *>(&e)) {
    return ErrorSource::popart;
  }
  if (dynamic_cast<const poplar::poplar_error *>(&e)) {
    return ErrorSource::poplar;
  }
  if (dynamic_cast<const poputil::poplibs_error *>(&e)) {
    return ErrorSource::poplibs;
  }
  return ErrorSource::unknown;
}

} // namespace popart
