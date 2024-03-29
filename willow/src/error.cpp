// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <exception>
#include <sstream>
#include <string>
#include <poplar/exceptions.hpp>
#include <poputil/exceptions.hpp>
#include <popart/error.hpp>
#include <popart/logging.hpp>

#include "popart/erroruid.hpp"

#ifdef POPART_USE_STACKTRACE
#include <boost/stacktrace.hpp>
#endif

namespace popart {

const std::string &error::stackreport() const { return _stack; }

std::string error::prependUid(ErrorUid uid, const std::string &s) {
  return "POPART" + std::to_string(static_cast<int>(uid)) + ": " + s;
}

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
    stackreport << "[" << i << "] " << logging::escape(st[i].name()) << "\n";
  }

  if (stackreport.tellp() > 0) {
    _stack = stackreport.str();
    oss << "\n\n" << _stack << "\n\n";
  }
#endif

  logging::err("{}", oss.str());
}

ErrorSource getErrorSource(const std::exception &e) {
  logging::warn(
      "getErrorSource is deprecated and will be removed in a future release.");
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
