// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <popart/error.hpp>
#include <popart/logging.hpp>

#include <poplar/exceptions.hpp>
#include <poputil/exceptions.hpp>

#ifdef POPART_USE_STACKTRACE
#include <boost/stacktrace.hpp>
#endif

namespace popart {

void error::logMessage() {
  std::ostringstream oss;
  oss << what();

#ifdef POPART_USE_STACKTRACE
  static constexpr size_t numFramesToSkip = 3;
  static constexpr size_t maxDepth        = 8;
  boost::stacktrace::stacktrace st(numFramesToSkip, maxDepth);
  std::ostringstream stackreport;

  for (size_t i = 0; i < st.size(); i++) {
    if (st[i].name().empty()) {
      // empty name -> truncate stack report
      break;
    }
    stackreport << "[" << i << "] " << st[i].name() << "\n";
  }

  if (stackreport.tellp() > 0) {
    oss << "\n\n" << stackreport.str() << "\n\n";
  }
#endif

  logging::err(oss.str());
}

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
