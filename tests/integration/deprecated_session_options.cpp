// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE DeprecatedSessionOptions

#include <boost/filesystem.hpp>
#include <boost/test/unit_test.hpp>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <popart/sessionoptions.hpp>

#include "popart/logging.hpp"
#include "popart/names.hpp"

using namespace popart;

// create a temporary file name to write logs to
std::string logger_output_file = (boost::filesystem::temp_directory_path() /
                                  boost::filesystem::unique_path())
                                     .native();

// fixture to redirect logs to file above
// This is done globally as logger destination gets initialized exactly once
struct Fixture {
  Fixture() { setenv("POPART_LOG_DEST", logger_output_file.c_str(), true); }
  ~Fixture() { /* Run on tear down */
  }
};
BOOST_GLOBAL_FIXTURE(Fixture);

BOOST_AUTO_TEST_CASE(TestDeprecatedPrefetchBufferingDepthMapOption) {
  // make sure to clear log file
  (std::ofstream(logger_output_file));

  // set up SessionOptions
  auto opts = SessionOptions();
  std::map<TensorId, unsigned> tensor_map{{"some_tensor_id", 1}};
  opts.prefetchBufferingDepthMap = tensor_map;
  TensorId any_tensor_id         = "any_tensor_id";

  // get buffering depth the first time
  opts.getBufferingDepth(any_tensor_id, false);

  // check that prefetchBufferingDepthMap was moved to bufferingDepthMap
  BOOST_CHECK_EQUAL(tensor_map, opts.bufferingDepthMap);
  std::map<TensorId, unsigned> empty_map;
  BOOST_CHECK(opts.prefetchBufferingDepthMap == empty_map);

  // flush logging output
  logging::flush(logging::Module::none);

  // check that we got a warning
  std::ifstream warning_stream(logger_output_file);
  std::stringstream warning;
  warning << warning_stream.rdbuf();
  BOOST_CHECK(
      warning.str().find("prefetchBufferingDepthMap has been deprecated") !=
      std::string::npos);

  // clear log output file
  (std::ofstream(logger_output_file));

  // get buffering depth a second time
  opts.getBufferingDepth(any_tensor_id, false);

  // flush logging output
  logging::flush(logging::Module::none);

  // check that no warning gets printed the second time
  warning_stream = std::ifstream(logger_output_file);
  warning.str(std::string());
  warning << warning_stream.rdbuf();
  BOOST_CHECK(warning.str() == "");
}

BOOST_AUTO_TEST_CASE(TestDeprecatedDefaultPrefetchBufferingDepthOption) {
  // make sure to clear log file
  (std::ofstream(logger_output_file));

  // set up SessionOptions
  auto opts                          = SessionOptions();
  unsigned bufferingDepth            = 3;
  opts.defaultPrefetchBufferingDepth = bufferingDepth;
  TensorId any_tensor_id             = "any_tensor_id";

  // get buffering depth the first time
  opts.getBufferingDepth(any_tensor_id, false);

  // check that prefetchBufferingDepthMap was moved to bufferingDepthMap
  BOOST_CHECK_EQUAL(bufferingDepth, opts.defaultBufferingDepth);
  auto initialDefaultPrefetchBufferingDepthValue = 111122;
  BOOST_CHECK(opts.defaultPrefetchBufferingDepth ==
              initialDefaultPrefetchBufferingDepthValue);

  // flush logging output
  logging::flush(logging::Module::none);

  // check that we got a warning
  std::ifstream warning_stream(logger_output_file);
  std::stringstream warning;
  warning << warning_stream.rdbuf();
  BOOST_CHECK(
      warning.str().find("defaultPrefetchBufferingDepth has been deprecated") !=
      std::string::npos);

  // clear log output file
  (std::ofstream(logger_output_file));

  // get buffering depth a second time
  opts.getBufferingDepth(any_tensor_id, false);

  // flush logging output
  logging::flush(logging::Module::none);

  // check that no warning gets printed the second time
  warning_stream = std::ifstream(logger_output_file);
  warning.str(std::string());
  warning << warning_stream.rdbuf();
  BOOST_CHECK(warning.str() == "");
}
