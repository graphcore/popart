// Copyright (c) 2018, Graphcore Ltd, All rights reserved

#include <poponnx/logging.hpp>

#include <spdlog/fmt/fmt.h>
#include <spdlog/sinks/ansicolor_sink.h>
#include <spdlog/sinks/file_sinks.h>
#include <spdlog/spdlog.h>

#include <boost/exception/diagnostic_information.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include <iostream>
#include <string>

namespace poponnx {
namespace logging {

namespace {

// Check our enums match (incase spdlog changes under us)
static_assert(static_cast<spdlog::level::level_enum>(Level::Trace) ==
                  spdlog::level::trace,
              "Logging enum mismatch");
static_assert(static_cast<spdlog::level::level_enum>(Level::Off) ==
                  spdlog::level::off,
              "Logging enum mismatch");

// Translate to a speedlog log level.
spdlog::level::level_enum translate(Level l) {
  return static_cast<spdlog::level::level_enum>(l);
}

// Stores the logging object needed by spdlog.
class LoggingContext {

public:
  static std::shared_ptr<spdlog::logger> getLogger(Module m);

  static void setDefaultLogLevel(Level l) { instance().defaultLevel = l; }

private:
  LoggingContext();

  static LoggingContext &instance() {
    // This avoids the static initialisation order fiasco, but doesn't solve the
    // deinitialisation order. Who logs in destructors anyway?
    static LoggingContext loggingContext;
    return loggingContext;
  }

  Level defaultLevel;
  boost::property_tree::ptree loggingConfig;
  std::shared_ptr<spdlog::sinks::sink> sink;
  std::map<Module, std::shared_ptr<spdlog::logger>> loggers;
};

Level logLevelFromString(const std::string &level) {

  if (level == "TRACE")
    return Level::Trace;
  if (level == "DEBUG")
    return Level::Debug;
  if (level == "INFO")
    return Level::Info;
  if (level == "WARN")
    return Level::Warn;
  if (level == "ERR")
    return Level::Err;
  if (level == "CRITICAL")
    return Level::Critical;
  if (level == "OFF")
    return Level::Off;

  return Level::Off;
}

Module moduleFromString(const std::string &module) {
  if (module == "session")
    return Module::session;
  if (module == "ir")
    return Module::ir;
  if (module == "devicex")
    return Module::devicex;
  if (module == "poponnx")
    return Module::poponnx;
  if (module == "transform")
    return Module::transform;
  if (module == "pattern")
    return Module::pattern;

  return Module::none;
}

std::string moduleName(const Module m) {
  std::string prefix = "poponnx:";
  std::string module;
  switch (m) {
  case Module::session:
    module = "session";
    break;
  case Module::ir:
    module = "ir";
    break;
  case Module::devicex:
    module = "devicex";
    break;
  case Module::poponnx:
    module = "poponnx";
    break;
  case Module::transform:
    module = "transform";
    break;
  case Module::pattern:
    module = "pattern";
    break;
  case Module::none:
  default:
    module = "<unknown>";
    break;
  }

  return prefix + module;
}

const char *defaultLoggerDestination = "stderr";
const char *defaultLoggerLevel       = "OFF";

LoggingContext::LoggingContext() {
  auto POPONNX_LOG_DEST   = std::getenv("POPONNX_LOG_DEST");
  auto POPONNX_LOG_LEVEL  = std::getenv("POPONNX_LOG_LEVEL");
  auto POPONNX_LOG_CONFIG = std::getenv("POPONNX_LOG_CONFIG");

  // Get logging output from the POPONNX_LOG_DEST environment variable.
  // The valid options are "stdout", "stderr", or if it is neither
  // of those it is treated as a filename. The default is stderr.
  std::string logDest =
      POPONNX_LOG_DEST ? POPONNX_LOG_DEST : defaultLoggerDestination;

  // Get logging level from OS ENV. The default level is off.
  defaultLevel = logLevelFromString(POPONNX_LOG_LEVEL ? POPONNX_LOG_LEVEL
                                                      : defaultLoggerLevel);

  if (POPONNX_LOG_CONFIG) {
    try {
      boost::property_tree::read_json(POPONNX_LOG_CONFIG, loggingConfig);
    } catch (const boost::exception &) {
      std::cerr << "Error reading log configuration file: "
                << boost::current_exception_diagnostic_information()
                << std::endl;
      throw;
    }
  }

  // Create the logging sink based on the logDest setting
  if (logDest == "stdout") {
    sink = std::make_shared<spdlog::sinks::ansicolor_stdout_sink_mt>();
  } else if (logDest == "stderr") {
    sink = std::make_shared<spdlog::sinks::ansicolor_stderr_sink_mt>();
  } else {
    try {
      sink =
          std::make_shared<spdlog::sinks::simple_file_sink_mt>(logDest, true);
    } catch (const spdlog::spdlog_ex &e) {
      // Should we be throwing an poponnx exception?
      std::cerr << "Error opening log file: " << e.what() << std::endl;
      throw;
    }
  }

  // Set the pattern of the logging output
  spdlog::set_pattern("%T.%e %t [%n:%L] %v");
}

// Find the logger is already created, else create a new logger
std::shared_ptr<spdlog::logger> LoggingContext::getLogger(Module m) {

  LoggingContext &instance = LoggingContext::instance();

  auto it = instance.loggers.find(m);

  if (it != instance.loggers.end()) {
    return it->second;
  } else {
    // create the logger for module m
    auto logger =
        std::make_shared<spdlog::logger>(moduleName(m), instance.sink);

    // read the logging level if defined in the logging config file
    Level level = instance.defaultLevel;
    boost::optional<std::string> levelCfg =
        instance.loggingConfig.get_optional<std::string>(moduleName(m));
    if (levelCfg)
      level = logLevelFromString(*levelCfg);

    logger->set_level(translate(level));

    // save the logger
    instance.loggers[m] = logger;

    return logger;
  }
}

} // namespace

// Implementation of public logging api

void configure(const std::map<std::string, std::string> &config) {
  for (auto p : config) {

    std::string moduleName = p.first;
    if (moduleName == "all") {
      Level l = logLevelFromString(p.second);
      LoggingContext::setDefaultLogLevel(l);
    } else {
      Module m = moduleFromString(moduleName);
      Level l  = logLevelFromString(p.second);
      setLogLevel(m, l);
    }
  }
}

void log(Module m, Level l, std::string &&msg) {
  LoggingContext::getLogger(m)->log(translate(l), msg);
}

bool shouldLog(Module m, Level l) {
  return LoggingContext::getLogger(m)->should_log(translate(l));
}

void setLogLevel(Module m, Level l) {
  LoggingContext::getLogger(m)->set_level(translate(l));
}

void flush(Module m) { LoggingContext::getLogger(m)->flush(); }

} // namespace logging
} // namespace poponnx
