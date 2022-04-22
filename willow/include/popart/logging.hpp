// Copyright (c) 2018 Graphcore Ltd. All rights reserved.

#ifndef GUARD_LOGGING_HPP
#define GUARD_LOGGING_HPP

#include <cstddef>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

/*
 *
 * This is a simple logging system for Popart based on Poplar based on spdlog.
 * The easiest way to use it is to simply call `logging::<module>::<level>()`
 * <module> is one of the modules & <level> is one of trace, debug, info,
 * warn, err or critical. For example:
 *
 *   #include <popart/logging.hpp>
 *
 *   void foo(int i) {
 *     logging::session::info("foo({}) called", i);
 *   }
 *
 * logging can be configured by the methods below, or by environment
 * variables, eg
 * POPART_LOG_LEVEL=ERR
 * POPART_LOG_DEST=Mylog.txt
 * POPART_LOG_CONFIG=log.cfg
 *
 * Formatting is done using the `fmt` library. It supports {}-style and %-style
 * format specification strings. See https://github.com/fmtlib/fmt for details.

 * To add a new logging module
 * 1. Add an entry to the Module enum
 * 2. Add a MAKE_MODULE_TEMPLATE macro call
 * 3. Add the entry to the string conversion functions in the .cpp file
 * 4. Update the ONNX webpage to describe the new module
 *    (https://phabricator.sourcevertex.net/w/onnx/)
 *
 * \endverbatim
 */

namespace popart {
namespace logging {

// Popart uses spdlog for logging. Originally this was exposed by this module as
// used in the rest of the popart. However this introduced a dependency that
// users of popart needed to resolve. The following internal namespace provide
// 'just enough' of the spdlog interface to allow use to hide the spdlog in the
// .cpp file and not expose the dependency.

namespace internal {
struct Value {

  Value() = default;
  Value(const Value &v) : custom_value(v.custom_value) {}
  ~Value() {}

  typedef std::ostream &(*ValueFormatFunc)(std::ostream &os, const void *t);

  struct CustomValue {
    const void *value      = nullptr;
    ValueFormatFunc format = nullptr;
  };

  CustomValue custom_value;
};

template <typename T>
std::ostream &formatValue(std::ostream &os, const void *arg) {
  const T *t = static_cast<const T *>(arg);
  os << (*t);
  return os;
}

template <typename T> Value MakeValue(const T &value) {
  Value v;
  v.custom_value.value  = &value;
  v.custom_value.format = &formatValue<T>;
  return v;
}

template <std::size_t N> struct ArgArray {
  // Zero sized arrays are not allowed
  typedef Value Type[N + 1];

  template <typename T> static Value make(T &value) { return MakeValue(value); }

  template <typename T> static Value make(const T &value) {
    return MakeValue(value);
  }
};

std::string format(std::string ref, std::size_t numArgs, Value args[]);

} // namespace internal

// Replace '{' with '{{' and '}' with '}}'.
std::string escape(const std::string &ref);

enum class Level {
  Trace    = 0,
  Debug    = 1,
  Info     = 2,
  Warn     = 3,
  Err      = 4,
  Critical = 5,
  Off      = 6,
  N        = 7, // Number of levels
};

enum class Module {
  popart,    /// Generic popart module, used when the module is not passed
  session,   /// Session module
  ir,        /// Ir module
  devicex,   /// Devicex module
  transform, /// Transform module
  pattern,   /// Pattern module
  builder,   /// Builder module
  op,        /// Op module
  opx,       /// Opx module
  ces,       /// Const Expr module
  python,    /// The python module
  none       /// The undefined module
};

// configure the logging using a map of modules to level
void configure(const std::map<std::string, std::string> &config);

// Set the current log level to one of the above levels. The default
// log level is set by the POPART_LOG_LEVEL environment variable
// and is off by default.
void setLogLevel(Module m, Level l);

Level getLogLevel(Module m);

// Return true if the passed log level is currently enabled.
bool shouldLog(Module m, Level l);

// Flush the log. By default it is only flushed when the underlying libc
// decides to.
void flush(Module m);

// Log a message. You should probably use the MAKE_LOG_TEMPLATE macros
// instead, e.g. logging::debug("A debug message").
void log(Module m, Level l, const std::string &&msg);

// Custom exception throw if there is an error in the log string formating
struct FormatError : public std::runtime_error {
  FormatError(std::string reason) : std::runtime_error(reason) {}
};

// The magic happens here. The format method collect the variadic template
// arguments are passes them to the internal format function.
template <typename... Args>
std::string format(std::string ref, const Args &... args) {
  typedef internal::ArgArray<sizeof...(Args)> ArgArray;
  typename ArgArray::Type array{ArgArray::template make(args)...};
  return internal::format(ref, sizeof...(Args), array);
}

// Reiimplementation of spdlog join function.
template <typename It> std::string join(It begin, It end, std::string sep) {
  std::stringstream ss;

  It it = begin;
  if (it != end) {
    ss << *it++;
    while (it != end) {
      ss << sep;
      ss << *it++;
    }
  }
  return ss.str();
}

// Log a formatted message. This uses the `fmt` C++ library for formatting.
// See https://github.com/fmtlib/fmt for details. You should probably use
// the MAKE_LOG_TEMPLATE macros instead, e.g.
// logging::session::debug("The answer is: {}", 42).
template <typename... Args>
void log(Module m, Level l, const char *s, const Args &... args) {
  // Avoid formatting if the logging is disabled anyway.
  if (shouldLog(m, l)) {
    log(m, l, format(s, args...));
  }
}

// Create a bit of syntactic sugar which allows log statements

// of the form logging::debug("Msg").
// where session if the name of the log module and debug if the
// logging level
#define MAKE_LOG_TEMPLATE(fnName, lvl)                                         \
  template <typename... Args>                                                  \
  inline void fnName(const std::string &s, const Args &... args) {             \
    log(Module::popart,                                                        \
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

#undef MAKE_LOG_TEMPLATE

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
MAKE_MODULE_TEMPLATE(transform)
MAKE_MODULE_TEMPLATE(pattern)
MAKE_MODULE_TEMPLATE(builder)
MAKE_MODULE_TEMPLATE(op)
MAKE_MODULE_TEMPLATE(opx)
MAKE_MODULE_TEMPLATE(ces)
MAKE_MODULE_TEMPLATE(python)

#undef MAKE_MODULE_LOG_TEMPLATE
#undef MAKE_MODULE_TEMPLATE

} // namespace logging
} // namespace popart

#endif // GUARD_LOGGING_HPP
