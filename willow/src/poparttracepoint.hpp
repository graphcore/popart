// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_POPARTTRACEPOINT_HPP
#define GUARD_NEURALNET_POPARTTRACEPOINT_HPP

// Would liked to have used string_view but that requires C++17
// Instead create a wrapper for pointer & length
// #include <string_view>

#include <cstddef>
#include <pvti/pvti.hpp>
#include <string>

namespace popart {

// Wrapper class of the pvti Tracepoint for Poplar

struct string_view {
  const char *ptr;
  size_t len;
};

class PopartTracepoint : public pvti::Tracepoint {
  static pvti::TraceChannel tracePopart;

public:
  PopartTracepoint(const std::string &traceLabel)
      : pvti::Tracepoint(&PopartTracepoint::tracePopart, traceLabel) {}

  PopartTracepoint(const char *traceLabel)
      : pvti::Tracepoint(&PopartTracepoint::tracePopart, traceLabel) {}

  PopartTracepoint(const string_view traceLabel)
      : pvti::Tracepoint(&PopartTracepoint::tracePopart,
                         traceLabel.ptr,
                         traceLabel.len) {}

  ~PopartTracepoint() = default;
};

constexpr string_view format_pretty_function(const char *s) {
  // First find the opening brackets for the arguments
  char const *b = s;
  while (*b != '(' && *b != '\0') {
    b++;
  }

  // Search backwards for the first space
  char const *c = b;
  while (*c != ' ' && c != s) {
    c--;
  }

  // c can equal s if the function has no return type i.e. constructors.
  if (c == s) {
    return {s, static_cast<size_t>(b - s)};
  } else {
    // +1 as c points to the ' '
    return {c + 1, static_cast<size_t>(b - (c + 1))};
  }
}

#define __POPART_FUNCTION_NAME__ format_pretty_function(__PRETTY_FUNCTION__)

#define POPART_TRACEPOINT()                                                    \
  PopartTracepoint __pt(format_pretty_function(__PRETTY_FUNCTION__))

} // namespace popart

#endif // GUARD_NEURALNET_POPARTTRACEPOINT_HPP
