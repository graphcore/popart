// Copyright (c) 2018, Graphcore Ltd, All rights reserved

#ifndef GUARD_NEURALNET_ERROR_HPP
#define GUARD_NEURALNET_ERROR_HPP

#include <boost/stacktrace.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <poponnx/logging.hpp>

namespace poponnx {

/**
 * Exception class for poponnx
 */
class error : public std::runtime_error {

  // A type to ensure that the variadic constructors do not get called
  struct _empty {};

  // Constructors that do not throw exception, used in the case that the
  // fmt::format function throws an exception
  explicit error(const _empty &, const char *s) : std::runtime_error(s) {
    logging::err(what());
  }

  explicit error(const _empty &, const std::string &s) : std::runtime_error(s) {
    logging::err(what());
  }

public:
  /// Variadic constructor for error which allows the user to use a fmt string
  /// for the message. As the fmt::format function can throw an exception itself
  /// and it is used in the initilization list we have to use the unusally C++
  /// syntax to catch that exception and convert it to a poponnx exception
  ///
  /// throw error("This is an error reason {}", 42);

  template <typename... Args>
  explicit error(const char *s, const Args &... args) try : std
    ::runtime_error(fmt::format(s, args...)) {
      logging::err(what());
      std::stringstream ss;
      ss << boost::stacktrace::stacktrace();
      logging::err(ss.str());
    }
  catch (const fmt::FormatError &e) {
    std::string reason =
        std::string("Poponnx exception format error ") + std::string(e.what());
    error _e(_empty(), reason);
    throw _e;
  }

  template <typename... Args>
  explicit error(const std::string &s, const Args &... args) try : std
    ::runtime_error(fmt::format(s, args...)) {
      logging::err(what());
      std::stringstream ss;
      ss << boost::stacktrace::stacktrace();
      logging::err(ss.str());
    }
  catch (const fmt::FormatError &e) {
    std::string reason =
        std::string("Poponnx exception format error:") + std::string(e.what());
    throw error(_empty(), reason);
  }
};

enum ErrorSource {
  poponnx = 0,
  poplar  = 1,
  poplibs = 2,
  unknown = 3,
};

ErrorSource getErrorSource(const std::exception &e);

} // namespace poponnx

#endif
