// Copyright (c) 2018 Graphcore Ltd. All rights reserved.

#ifndef POPART_WILLOW_INCLUDE_POPART_ERROR_HPP_
#define POPART_WILLOW_INCLUDE_POPART_ERROR_HPP_

#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <popart/erroruid.hpp>
#include <popart/logging.hpp>

namespace popart {

/**
 * Exception class for popart
 */
class error : public std::runtime_error {

  // A type to ensure that the variadic constructors do not get called
  struct _empty {};

  // Constructors that do not throw exception, used in the case that the
  // fmt::format function throws an exception
  explicit error(const _empty &, const char *s) : std::runtime_error(s) {
    logMessage();
  }

  explicit error(const _empty &, const std::string &s) : std::runtime_error(s) {
    logMessage();
  }

public:
  /// Variadic constructor for error which allows the user to use a fmt string
  /// for the message.
  ///
  /// throw error("This is an error reason {}", 42);
  template <typename... Args>
  explicit error(const char *s, const Args &... args)
      : std ::runtime_error(formatMessage(s, args...)), uid_(ErrorUid::E0) {
    logMessage();
  }

  template <typename... Args>
  explicit error(const std::string &s, const Args &... args)
      : std ::runtime_error(formatMessage(s, args...)), uid_(ErrorUid::E0) {
    logMessage();
  }

  // Constructor that accepts a unique error ID as the first argument. For
  // example:
  //     throw error(ErrorUid::Foo, "This is an error reason {}", 42);
  template <typename... Args>
  explicit error(ErrorUid uid, const char *s, const Args &... args)
      : std::runtime_error(formatMessage(prependUid(uid, s), args...)),
        uid_(uid) {
    logMessage();
  }

  template <typename... Args>
  explicit error(ErrorUid uid, const std::string &s, const Args &... args)
      : std::runtime_error(formatMessage(prependUid(uid, s), args...)),
        uid_(uid) {
    logMessage();
  }

  const std::string &stackreport() const;

  ErrorUid uid() const { return uid_; }

private:
  /// As the fmt::format function can throw an exception itself we catch
  /// the FormatError exception here and convert it to a popart exception.
  template <typename... Args>
  static std::string formatMessage(const Args &... args) {
    try {
      return logging::format(args...);
    } catch (const logging::FormatError &e) {
      std::string reason =
          std::string("Popart exception format error ") + std::string(e.what());
      error _e(_empty(), reason);
      throw _e;
    }
  }

  static std::string prependUid(ErrorUid, const std::string &);

  /// Log the exception message
  //  Optionally appends a stacktrace depending on the build configuration.
  void logMessage();

  std::string _stack;
  ErrorUid uid_;
};

/**
 * Exception class specific to internal errors
 * This should be used as an assert; for states where the user should not have
 * been able to create.
 */
class internal_error : public error {
public:
  using error::error;
};

/**
 * Exception class specific to errors that occur when running a model. For
 * example, this error could be thrown when a user-implemented IStepIO callback
 * doesn't return any data.
 *
 * NOTE: This is different from a C++ runtime error.
 */
class runtime_error : public error {
public:
  using error::error;
};

enum class ErrorSource {
  popart = 0,
  popart_internal,
  poplar,
  poplibs,
  unknown,
};

// A specialization of the popart error exception for the case when the device
// prepare call fails due to lack of memory

class memory_allocation_err : public error {

public:
  memory_allocation_err(const std::string &info) : error("{}", info) {}

  virtual std::unique_ptr<memory_allocation_err> clone() const = 0;
  virtual std::string getSummaryReport() const                 = 0;
  virtual std::string getProfilePath() const                   = 0;
};

ErrorSource getErrorSource(const std::exception &e);

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_ERROR_HPP_
