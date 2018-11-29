// Copyright (c) 2018, Graphcore Ltd, All rights reserved

#ifndef GUARD_LOGGING_HPP
#define GUARD_LOGGING_HPP

#include <map>
#include <spdlog/fmt/fmt.h>
#include <spdlog/fmt/ostr.h>
#include <string>

/// This is a simple logging system for Poponnx based on Poplar based on spdlog.
/// The easiest way to use it is to simply call `logging::<module>::<level>()`
/// <module> is one of the modules & <level> is one of trace, debug, info,
/// warn, err or critical. For example:
///
///   #include <poponnx/logging.hpp>
///
///   void foo(int i) {
///     logging::session::info("foo({}) called", i);
///   }
///
/// logging can be configured by the methods below, or by environment
/// variables, eg
/// POPONNX_LOG_LEVEL=ERR
/// POPONNX_LOG_DEST=Mylog.txt
/// POPONNX_LOG_CONFIG=log.cfg
///
/// Formatting is done using the `fmt` library. It supports {}-style and %-style
/// format specification strings. See https://github.com/fmtlib/fmt for details.

/// To add a new logging module
/// 1. Add an entry to the Module enum
/// 2. Add a MAKE_MODULE_TEMPLATE macro call
/// 3. Add the entry to the string conversion functions in the .cpp file
/// 4. Update the ONNX webpage to describe the new module
///    (https://phabricator.sourcevertex.net/w/onnx/)

namespace poponnx {
namespace logging {

enum class Level {
  Trace    = 0,
  Debug    = 1,
  Info     = 2,
  Warn     = 3,
  Err      = 4,
  Critical = 5,
  Off      = 6,
};

enum class Module {
  poponnx, /// Generic poponnx module, used when the module is not passed
  session, /// Session module
  ir,      /// Ir module
  devicex, /// Devicex module
  none     /// The undefined module
};

// configure the logging using a map of modules to level
void configure(const std::map<std::string, std::string> &config);

// Set the current log level to one of the above levels. The default
// log level is set by the POPONNX_LOG_LEVEL environment variable
// and is off by default.
void setLogLevel(Module m, Level l);

// Return true if the passed log level is currently enabled.
bool shouldLog(Module m, Level l);

// Flush the log. By default it is only flushed when the underlying libc
// decides to.
void flush(Module m);

// Log a message. You should probably use the MAKE_LOG_TEMPLATE macros
// instead, e.g. logging::debug("A debug message").
void log(Module m, Level l, std::string &&msg);

// Log a formatted message. This uses the `fmt` C++ library for formatting.
// See https://github.com/fmtlib/fmt for details. You should probably use
// the MAKE_LOG_TEMPLATE macros instead, e.g.
// logging::session::debug("The answer is: {}", 42).
template <typename... Args>
void log(Module m, Level l, const char *s, const Args &... args) {
  // Avoid formatting if the logging is disabled anyway.
  if (shouldLog(m, l)) {
    log(m, l, fmt::format(s, args...));
  }
}

// Create a bit of syntactic sugar which allows log statements

// of the form logging::debug("Msg").
// where session if the name of the log module and debug if the
// logging level
#define MAKE_LOG_TEMPLATE(fnName, lvl)                                         \
  template <typename... Args>                                                  \
  inline void fnName(const std::string &s, const Args &... args) {             \
    log(Module::poponnx,                                                       \
        Level::lvl,                                                            \
        s.c_str(),                                                             \
        std::forward<const Args>(args)...);                                    \
  }

MAKE_LOG_TEMPLATE(trace, Trace)
MAKE_LOG_TEMPLATE(debug, Debug)
MAKE_LOG_TEMPLATE(info, Info)
MAKE_LOG_TEMPLATE(warn, Warn)
MAKE_LOG_TEMPLATE(err, Err)
MAKE_LOG_TEMPLATE(crit, Critical)

// of the form logging::session::debug("Msg").
// where session if the name of the log module and debug if the
// logging level
#define MAKE_MODULE_LOG_TEMPLATE(fnName, module, lvl)                          \
  template <typename... Args>                                                  \
  inline void fnName(const std::string &s, const Args &... args) {             \
    log(Module::module,                                                        \
        Level::lvl,                                                            \
        s.c_str(),                                                             \
        std::forward<const Args>(args)...);                                    \
  }

#define MAKE_MODULE_TEMPLATE(MODULE)                                           \
  namespace MODULE {                                                           \
  MAKE_MODULE_LOG_TEMPLATE(trace, MODULE, Trace)                               \
  MAKE_MODULE_LOG_TEMPLATE(debug, MODULE, Debug)                               \
  MAKE_MODULE_LOG_TEMPLATE(info, MODULE, Info)                                 \
  MAKE_MODULE_LOG_TEMPLATE(warn, MODULE, Warn)                                 \
  MAKE_MODULE_LOG_TEMPLATE(err, MODULE, Err)                                   \
  MAKE_MODULE_LOG_TEMPLATE(crit, MODULE, Critical)                             \
  inline void flush() {                                                        \
    Module m = Module::MODULE;                                                 \
    flush(m);                                                                  \
  }                                                                            \
  inline void setLogLevel(Level l) { setLogLevel(Module::MODULE, l); }         \
  inline bool isEnabled(Level l) { return shouldLog(Module::MODULE, l); }      \
  }

// The definition of the logging modules
MAKE_MODULE_TEMPLATE(session)
MAKE_MODULE_TEMPLATE(ir)
MAKE_MODULE_TEMPLATE(devicex)

// Convenience macro to create a log entry prefixed with function name e.g.:
//    void someFunc(int i) {
//      FUNC_LOGGER(session::info, " with i := {}", i);
//    }
// Then the log entry would be something like:
// 14:30:31.00 [I] void someFunc(int): with i := 42
// NOTE: Because of the limitations of __VA_ARGS__ this log entry must have at
// least one parameter.
#define FUNC_LOGGER(lvl, fmtStr, ...)                                          \
  logging::lvl("{}: " fmtStr, __PRETTY_FUNCTION__, __VA_ARGS__)

} // namespace logging
} // namespace poponnx

#endif // GUARD_LOGGING_HPP
